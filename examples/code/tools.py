"""
Tool sandbox module for safe code execution and tool management.

This module provides:
- PythonSandbox: Safe Python code execution environment
- ToolRegistry: Tool registration and execution management
- Memory management utilities
"""

import asyncio
import gc
import os
import re
import subprocess
import yaml
import time
import tempfile
import logging
import psutil
import requests
import resource
import json
import threading
import traceback
import tempfile
from uuid import uuid4
from contextlib import contextmanager
import concurrent.futures
from typing import Any, Optional, Callable, TypeVar


DEFAULT_TIMEOUT = 10  # Default compile and run timeout
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1
API_TIMEOUT = 10

PY_IMPORTS = yaml.safe_load(open("examples/code/deps/header.yaml"))['python']
logger = logging.getLogger(__name__)
# Configuration for tool execution
TOOL_CONFIGS = {
    "max_turns": 16,
    "max_tool_calls": 16,
    "tool_concurrency": 32,  # Aggressive: 32 concurrent processes
    # Python interpreter settings
    "python_timeout": 120,  # 2 minutes for complex calculations
    "python_memory_limit": "4GB",  # 4GB per Python process
    "python_cpu_limit": 1,
    # Memory management settings
    "max_memory_usage": 12288,  # 12GB total (75% of 16GB)
    "cleanup_threshold": 6144,  # 6GB
    "aggressive_cleanup_threshold": 3072,  # 3GB
    "force_cleanup_threshold": 9216,  # 9GB
}

# Global semaphore for controlling concurrent tool executions
SEMAPHORE = asyncio.Semaphore(TOOL_CONFIGS["tool_concurrency"])


def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def cleanup_memory():
    """Force garbage collection to free memory"""
    gc.collect()


def aggressive_cleanup_memory():
    """More aggressive memory cleanup"""
    # Force multiple garbage collection cycles
    for _ in range(3):
        gc.collect()

    # Clear Python's internal caches
    import sys

    # Note: sys.intern doesn't have a clear method, so we skip this
    # Clear module cache if possible
    if hasattr(sys, "modules"):
        # Don't clear all modules, but clear some common ones that might cache data
        modules_to_clear = ["numpy", "pandas", "matplotlib", "scipy"]
        for module_name in modules_to_clear:
            if module_name in sys.modules:
                module = sys.modules[module_name]
                if hasattr(module, "clear_cache"):
                    module.clear_cache()


def check_and_cleanup_memory():
    """Check memory usage and perform appropriate cleanup"""
    current_memory = get_memory_usage()

    if current_memory > TOOL_CONFIGS["force_cleanup_threshold"]:
        # Force aggressive cleanup
        aggressive_cleanup_memory()
        return f"Warning: High memory usage ({current_memory:.1f}MB), performed aggressive cleanup"
    elif current_memory > TOOL_CONFIGS["cleanup_threshold"]:
        # Normal cleanup
        cleanup_memory()
        return f"Info: Memory usage ({current_memory:.1f}MB), performed cleanup"
    elif current_memory > TOOL_CONFIGS["aggressive_cleanup_threshold"]:
        # Light cleanup
        gc.collect()
        return f"Info: Memory usage ({current_memory:.1f}MB), performed light cleanup"

    return None


class PythonSandbox:
    """Python code sandbox, provides safe code execution environment"""

    def __init__(self, timeout: int = 10, memory_limit: str = "100MB"):
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.allowed_modules = {
            "math",
            "random",
            "datetime",
            "collections",
            "itertools",
            "functools",
            "operator",
            "statistics",
            "decimal",
            "fractions",
        }
        self.SUPPORTED_LANGUAGES = [
            "python",
            "cpp",
            "nodejs",
            "go",
            "go_test",
            "java",
            "php",
            "csharp",
            "bash",
            "typescript",
            "sql",
            "rust",
            "cuda",
            "lua",
            "R",
            "perl",
            "D_ut",
            "ruby",
            "scala",
            "julia",
            "pytest",
            "junit",
            "kotlin_script",
            "jest",
            "verilog",
            "python_gpu",
            "lean",
            "swift",
            "racket",
        ]

    def _check_code_safety(self, code: str) -> tuple[bool, str]:
        """Check code safety by scanning for dangerous patterns"""
        # Check for dangerous operations
        dangerous_patterns = [
            r"import\s+os",
            r"import\s+sys",
            r"import\s+subprocess",
            r"import\s+shutil",
            r"import\s+glob",
            r"import\s+pathlib",
            r"__import__",
            r"eval\s*\(",
            r"exec\s*\(",
            r"open\s*\(",
            r"file\s*\(",
            r"input\s*\(",
            r"raw_input\s*\(",
            r"compile\s*\(",
            r"execfile\s*\(",
            r"getattr\s*\(",
            r"setattr\s*\(",
            r"delattr\s*\(",
            r"hasattr\s*\(",
            r"globals\s*\(",
            r"locals\s*\(",
            r"vars\s*\(",
            r"dir\s*\(",
            r"type\s*\(",
            r"isinstance\s*\(",
            r"issubclass\s*\(",
            r"super\s*\(",
            r"property\s*\(",
            r"staticmethod\s*\(",
            r"classmethod\s*\(",
            r"__\w+__",  # double underscore methods
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Code contains dangerous pattern: {pattern}"

        # Check imported modules
        import_pattern = r"import\s+(\w+)"
        from_pattern = r"from\s+(\w+)"

        imports = re.findall(import_pattern, code)
        froms = re.findall(from_pattern, code)

        all_imports = set(imports + froms)
        for imp in all_imports:
            if imp not in self.allowed_modules:
                return False, f"Import of '{imp}' is not allowed"

        return True, "Code is safe"

    @contextmanager
    def _create_safe_environment(self):
        """Create safe execution environment with temporary directory"""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="python_sandbox_")

        try:
            # Create safe Python script
            script_path = os.path.join(temp_dir, "code.py")

            # Set environment variables
            env = os.environ.copy()
            env["PYTHONPATH"] = temp_dir
            env["PYTHONUNBUFFERED"] = "1"

            yield script_path, env, temp_dir

        finally:
            # Clean up temporary directory
            try:
                import shutil

                shutil.rmtree(temp_dir)
            except Exception:
                pass

    async def call_local_sandbox_api(
        self,
        code: str,
        stdin: Optional[str],
        compile_timeout: int,
        run_timeout: int,
        memory_limit_mb: int,
        language: str = "python",
    ) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        import subprocess
        import tempfile

        request_id = str(uuid4())
        log_prefix = f"[Request ID: {request_id}] "

        if language not in self.SUPPORTED_LANGUAGES:
            error_msg = f"{log_prefix} Unsupported language: {language}"
            logger.error(error_msg)
            return None, error_msg
        
        def prepare_workdir():
            parent_dir = "/tmp/firejail_code/"
            os.makedirs(parent_dir, exist_ok=True)
            workdir = tempfile.mkdtemp(prefix="fj_", dir=parent_dir)
            full_code = PY_IMPORTS + code
            script_path = os.path.join(workdir, "main.py")
            logger.info(f"{log_prefix}Writing code to {script_path}")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(full_code)
            return workdir
        
        try:
            workdir = await asyncio.to_thread(prepare_workdir)
                
            result = {
                "status": "unknown",
                "run_status": "unknown", 
                "run_result": None
            }

            cmd = [
                "firejail",
                f"--private={workdir}",              # 独立的工作目录
                "--rlimit-fsize=2m",                # 文件大小限制
                "--rlimit-nproc=32",
                "--rlimit-nofile=32",
                "--quiet",                          # 静默模式
                f"--timeout=00:00:{run_timeout}",   # 超时设置
                f"--whitelist={workdir}",           # 白名单工作目录
                language,                           # 语言（实际上是 python）
                "main.py",                          # 要执行的脚本
            ]

            logger.info(f"{log_prefix}Executing with firejail: {' '.join(cmd)}")
            
            # 执行代码
            run_result = {}
            start_time = time.monotonic()
            env = os.environ.copy()

            def _set_memory_limit(mb_limit: int):
                if mb_limit > 0:
                    limit_in_bytes = mb_limit * 4 * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (limit_in_bytes, limit_in_bytes))

            #env["OPENBLAS_NUM_THREADS"] = "1"
            if "PYTHONPATH" in env:
                del env["PYTHONPATH"]

            # 使用 asyncio.create_subprocess_exec 实现非阻塞子进程调用
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=workdir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE if stdin else None,
                env=env,
            )
            try:
                stdout_data, stderr_data = await asyncio.wait_for(
                    process.communicate(input=stdin.encode() if stdin else None),
                    timeout=run_timeout + 5
                )
                duration = time.monotonic() - start_time

                run_result["stdout"] = stdout_data.decode().strip()
                run_result["stderr"] = stderr_data.decode().strip()
                run_result["return_code"] = process.returncode
                run_result["execution_time"] = duration
                result = {"status": "unknown", "run_result": run_result}

                if process.returncode == 0:
                    result["status"] = "Success"
                    run_result["status"] = "Finished"
                elif process.returncode < 0:
                    import signal
                    signal_num = -process.returncode
                    signal_name = signal.Signals(signal_num).name

                    result["status"] = "Failed"
                    run_result["status"] = "MemoryLimitExceeded" 
                    run_result["stderr"] = f"Process was killed by signal {signal_num} ({signal_name}). This is often caused by exceeding a memory limit."
                else:
                    result["status"] = "Failed"
                    run_result["status"] = "Finished"

                result["run_result"] = run_result

                await asyncio.to_thread(lambda: __import__('shutil').rmtree(workdir, ignore_errors=True))
                logger.info(f"{log_prefix}Local sandbox execution completed successfully")
                return result, None
            except asyncio.TimeoutError:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                
                duration = time.monotonic() - start_time
                #logger.warning(f"{log_prefix}Process timed out after {duration:.2f}s")
                
                result = {
                    "status": "Failed", 
                    "run_result": {
                        "status": "TimeLimitExceeded",
                        "stderr": "TimeLimitExceeded",
                        "stdout": "",
                        "return_code": -1,
                        "execution_time": duration
                    }
                }
                
                return result, None
                
        except Exception as e:
            error_msg = f"{log_prefix}Unexpected error during local sandbox execution: {e}"
            logger.error(error_msg)
            error_details = traceback.format_exc() # 获取完整的错误堆栈 
            logger.error(error_details)
            result = {
                "status": "Failed",
                "run_status": "Error",
                "run_result": {
                    "status": "Error",
                    "stderr": f"An unexpected internal error occurred: {e}",
                    "stdout": "",
                    "return_code": -1,
                    "execution_time": 0
                }
            }
            return result, error_msg
        
    async def call_sandbox_api(
        self,
        sandbox_fusion_url: str,
        code: str,
        stdin: Optional[str],
        compile_timeout: int,
        run_timeout: int,
        memory_limit_mb: int,
        language: str = "python",
    ) -> tuple[Optional[dict[str, Any]], Optional[str]]: 
        """
        Calls the remote sandbox API to execute code with retry logic for Gateway Timeout,
        using increasing delay between retries. Logs internal calls with a unique ID.

        Args:
            sandbox_fusion_url: The URL of the sandbox fusion API.
            code: The code string to execute.
            stdin: The standard input string.
            compile_timeout: Compile timeout in seconds.
            run_timeout: Run timeout in seconds.
            language: The programming language of the code (e.g., "python", "cpp", "java"). Defaults to "python".

        Returns:
            A tuple (response_json, error_message).
            If successful, response_json is the API's returned JSON object, error_message is None.
            If failed after retries, response_json is None, error_message contains the error information.
        """
        request_id = str(uuid4())
        log_prefix = f"[Request ID: {request_id}] "  # <-- Create log prefix
        if language not in self.SUPPORTED_LANGUAGES:
            error_msg = f"{log_prefix}Unsupported language: {language}"
            logger.error(error_msg)
            return None, error_msg
        
        loop = asyncio.get_running_loop()

        def _sync_request():
            payload = json.dumps(
                {
                    "compile_timeout": compile_timeout,
                    "run_timeout": run_timeout,
                    "code": PY_IMPORTS + code,
                    "stdin": stdin,
                    "memory_limit_MB": memory_limit_mb,
                    "language": language, 
                    "files": {},
                    "fetch_files": [],
                }
            )
            headers = {"Content-Type": "application/json", "Accept": "application/json"}
            request_timeout = compile_timeout + run_timeout + API_TIMEOUT
            with requests.Session() as session:
                return session.post(
                    sandbox_fusion_url,
                    headers=headers,
                    data=payload,
                    timeout=request_timeout,
                )
            
        last_error = None  # Store the last error encountered

        for attempt in range(MAX_RETRIES):
            try:
                logger.info(
                    f"{log_prefix}Attempt {attempt + 1}/{MAX_RETRIES}: Calling sandbox API at {sandbox_fusion_url}"
                )  

                response = await loop.run_in_executor(None, _sync_request)

                # Check for Gateway Timeout (504) specifically for retrying
                if response.status_code == 504:
                    last_error = (
                        f"{log_prefix}API Request Error: Gateway Timeout (504) on attempt "
                        f"{attempt + 1}/{MAX_RETRIES}"
                    ) 
                    logger.warning(last_error)
                    if attempt < MAX_RETRIES - 1:  # Don't sleep after the last attempt
                        # Calculate increasing delay (e.g., 1s, 2s, 4s, ...) or (1s, 2s, 3s, ...)
                        # Simple linear increase: delay = INITIAL_RETRY_DELAY * (attempt + 1)
                        # Exponential backoff: delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                        delay = INITIAL_RETRY_DELAY * (attempt + 1)  # Using linear increase for simplicity
                        logger.info(f"{log_prefix}Retrying after {delay} seconds...")  # <-- Use internal log_prefix
                        await asyncio.sleep(delay)
                    continue  # Go to the next retry attempt

                # Check for other HTTP errors (e.g., 4xx, other 5xx)
                response.raise_for_status()

                # If successful (status code 2xx)
                logger.info(
                    f"{log_prefix}Sandbox API call successful on attempt {attempt + 1}"
                ) 
                return response.json(), None

            except requests.exceptions.RequestException as e:
                last_error = f"{log_prefix}API Request Error: {e}" 
                break 
            except json.JSONDecodeError as e:
                raw_response_text = response.text if "response" in locals() else "N/A"
                last_error = f"{log_prefix}API Response JSON Decode Error: {e}"  
                break  
            except Exception as e:
                last_error = str(e)
                # 一般错误不重试或根据需要重试
                if attempt < MAX_RETRIES -1 and isinstance(e, requests.exceptions.Timeout):
                     await asyncio.sleep(1)
                     continue
                break
            
        # If loop finishes without returning success, return the last recorded error
        logger.error(f"{log_prefix}Sandbox API call failed. Last error: {last_error}")  # <-- Use internal log_prefix
        # Return the error message without the prefix, as the caller doesn't need the internal ID
        # Ensure API call failure returns error message, leading to -1 in check_correctness
        return None, last_error.replace(log_prefix, "API Call Failed: ") if last_error else "API Call Failed after retries"


    async def _process_single_case(
        self,
        case_index: int,
        stdin_data: Any,
        expected_output: Any,
        sandbox_fusion_url: str,
        generation: str,
        timeout: int,
        memory_limit_mb: int,
        language: str,
        local_run: bool = False,
        concurrent_semaphore: Optional[threading.Semaphore] = None,
        fn_name: Optional[str] = None,
    ) -> tuple[int, dict[str, Any]]:
        """Helper function to process a single test case."""


        api_response = None
        error_msg = None

        current_generation_code = generation

        if fn_name and language == "python":
            # Wrapper assumes stdin_data is a JSON string for function arguments.
            current_generation_code = yaml.safe_load(open("examples/code/deps/wrapper.yaml"))['python']
        stdin = None if stdin_data is None else str(stdin_data)
        if local_run:
            api_response, error_msg = await self.call_local_sandbox_api(
                code=current_generation_code,
                stdin=stdin,
                compile_timeout=timeout,
                run_timeout=timeout,
                memory_limit_mb=memory_limit_mb,
                language=language,
            )
        else:
            try:
                if concurrent_semaphore:
                    # logger.debug(f"Case {case_index + 1}: Attempting to acquire semaphore.")
                    with concurrent_semaphore:
                        # logger.debug(f"Case {case_index + 1}: Semaphore acquired. Calling API.")
                        api_response, error_msg = await self.call_sandbox_api(
                            sandbox_fusion_url=sandbox_fusion_url,
                            code=current_generation_code,
                            stdin=stdin,
                            compile_timeout=timeout,
                            run_timeout=timeout,
                            memory_limit_mb=memory_limit_mb,
                            language=language,
                        )
                    # logger.debug(f"Case {case_index + 1}: Semaphore released.")
                else:
                    api_response, error_msg = await self.call_sandbox_api(
                        sandbox_fusion_url=sandbox_fusion_url,
                        code=current_generation_code,
                        stdin=stdin,
                        compile_timeout=timeout,
                        run_timeout=timeout,
                        memory_limit_mb=memory_limit_mb,
                        language=language,
                    )
            except Exception as e:
                error_msg = f"API Request Exception during check_correctness for case {case_index + 1}: {e}"
                logger.error(f"Case {case_index + 1}: {error_msg}")
                traceback.print_exc()

        metadata = {
            "case_index": case_index,
            "input": stdin,
            "expected_output": str(expected_output),
            "api_request_error": error_msg,
            "api_response": None,
            "status": "unknown",
            "stdout": None,
            "stderr": None,
            "exit_code": None,
            "duration": None,
            "compile_duration": None,
            "compile_stderr": None,
            "api_status": None,
            "compile_status": None,
            "run_status": None,
        }
        result_status = -1  # Default error: API request error or unknown sandbox error
        if error_msg:
            metadata["status"] = "api_error"
            result_status = -1  # API request itself failed (includes timeout after retries)
            logger.error(f"Case {case_index}: API error occurred: {error_msg}")
            # Log code and input only on error for brevity
            generation_to_log = generation[:200] + "..." if len(generation) > 200 else generation
            logger.error(f"Case {case_index}: code: {generation_to_log}")
            logger.error(f"Case {case_index}: input: {stdin}")
        elif api_response:
            # --- Add debug logging ---
            logger.debug(f"Case {case_index}: API Response: {api_response}")
            metadata["api_response"] = api_response
            metadata["api_status"] = api_response.get("status")
            compile_result = api_response.get("compile_result")
            run_result = api_response.get("run_result")

            # Extract compile information
            if compile_result:
                metadata["compile_status"] = compile_result.get("status")
                metadata["compile_duration"] = compile_result.get("execution_time")
                metadata["compile_stderr"] = compile_result.get("stderr")

            # Extract run information
            if run_result: 
                metadata["run_status"] = run_result.get("status")
                metadata["stdout"] = run_result.get("stdout")
                metadata["stderr"] = run_result.get("stderr")  # stderr during runtime
                metadata["exit_code"] = run_result.get("return_code")
                metadata["duration"] = run_result.get("execution_time")

            # --- Determine status based on API response ---
            api_status = metadata["api_status"]

            if api_status == "SandboxError":
                metadata["status"] = "sandbox_error"
                result_status = -1  # Internal sandbox error
            elif api_status == "Failed":
                # --- Add debug logging ---
                logger.debug(f"API returned Failed status. Response: {api_response}")
                logger.debug(f"Compile Result: {compile_result}")
                logger.debug(f"Run Result: {run_result}")
                # --- Check the logic here ---
                # Compile failed or timed out
                is_compile_error = compile_result and (
                    metadata["compile_status"] in ["Error", "TimeLimitExceeded"]
                    or (metadata["compile_status"] == "Finished" and compile_result.get("return_code") != 0)
                )
                if is_compile_error:
                    # Differentiate between compile_error and compile_timeout based on specific status
                    if metadata["compile_status"] == "TimeLimitExceeded":
                        metadata["status"] = "compile_timeout"
                    else:  # Includes Error and Finished but return_code != 0 cases
                        metadata["status"] = "compile_error"
                    result_status = -4
                # Run failed or timed out
                elif run_result:
                    # Modified condition: Check for TimeLimitExceeded OR (Finished with non-zero exit code) OR Error status
                    is_runtime_error = (
                        metadata["run_status"] == "TimeLimitExceeded"
                        or metadata["run_status"] == "Error"
                        or (metadata["run_status"] == "Finished" and run_result.get("return_code") != 0)
                        or metadata["run_status"] == "MemoryLimitExceeded"
                    )
                    if is_runtime_error:
                        if metadata["run_status"] == "TimeLimitExceeded":
                            metadata["status"] = "timeout"  # Runtime timeout
                            result_status = -3
                        elif metadata["run_status"] == "MemoryLimitExceeded":
                            metadata["status"] = "memory_limit_exceeded"  # Memory limit exceeded
                            result_status = -2
                        else:  # Includes Error and Finished with non-zero return_code
                            metadata["status"] = "runtime_error"
                            result_status = -2
                    else:
                        # Other Failed status with run_result, classify as unknown failure
                        logger.warning(f"Unknown run_status '{metadata['run_status']}' or state within Failed API status.")
                        metadata["status"] = "unknown_failure"
                        result_status = -1  # Default to -1
                else:
                    # Status is Failed but neither a clear compile error nor run_result exists
                    logger.warning("API status Failed but cannot determine specific error type (compile/run).")
                    metadata["status"] = "unknown_failure_state"
                    result_status = -1  # Default to -1
            elif api_status == "Success":
                # Run completed successfully, now check the answer
                if run_result and metadata["run_status"] == "Finished":
                    actual_output = metadata["stdout"] if metadata["stdout"] is not None else ""
                    # Note: Output might contain trailing newlines, need normalization
                    if str(actual_output).rstrip("\n") == str(expected_output).rstrip("\n"):
                        result_status = True
                        metadata["status"] = "success"
                    else:
                        result_status = False
                        metadata["status"] = "wrong_answer"
                else:
                    # Status is Success but run_result status is not Finished, this is unexpected
                    metadata["status"] = "unexpected_success_state"
                    result_status = -1  # Classify as unknown error
            else:
                # API returned an unknown top-level status
                logger.warning(f"Unknown API status received: {api_status}")
                metadata["status"] = f"unknown_api_status_{api_status}"
                result_status = -1  # Default to -1
        else:  # api_response is None and no error_msg (Should not happen with current call_sandbox_api logic)
            metadata["status"] = "unknown_api_state"
            result_status = -1
            logger.error(f"Case {case_index}: Unknown API state (no response and no error message).")
        return result_status, metadata


    async def check_correctness(
        self,
        sandbox_fusion_url: str,
        in_outs: Optional[dict],
        generation: str,
        timeout: int = DEFAULT_TIMEOUT,
        memory_limit_mb: int = 1024,
        language: str = "python",
        local_run: bool = False,
        concurrent_semaphore: Optional[threading.Semaphore] = None,
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        """
        Checks the correctness of code generation using the remote sandbox API,
        processing test cases concurrently.

        Args:
            sandbox_fusion_url: The URL of the sandbox fusion API.
            in_outs: Dictionary containing "inputs" and "outputs" lists.
            generation: The generated code string.
            timeout: Timeout for each test case (compile and run share this timeout).
            language: The programming language of the code.

        Returns:
            A tuple (results, metadata_list).
            results: A list containing the test result for each input/output pair
                    (True/False/-1 api/sandbox err, -2 runtime err, -3 timeout, -4 compile err).
                    Results are ordered corresponding to the inputs.
            metadata_list: A list containing metadata dictionaries for each test case,
                        ordered corresponding to the inputs.
        """
        logger.info(f"Starting correctness check for generation, {local_run=}")

        if not in_outs or "inputs" not in in_outs or "outputs" not in in_outs:
            logger.warning("Invalid in_outs format provided.")
            return [-1], [{"error": "Invalid input/output data"}]

        inputs = in_outs["inputs"]
        expected_outputs = in_outs["outputs"]
        fn_name = in_outs.get("fn_name", None)
        num_cases = len(inputs)
        results = [None] * num_cases  # Initialize with placeholders
        metadata_list = [None] * num_cases  # Initialize with placeholders
        breakpoint()
        if num_cases == 0:
            logger.warning("Empty inputs provided.")
            return [], []

        if len(inputs) != len(expected_outputs):
            logger.warning(f"Mismatch between number of inputs ({len(inputs)}) and outputs ({len(expected_outputs)}).")
            # Return error based on the number of inputs provided
            return [-1] * num_cases, [{"error": "Input/output count mismatch", "case_index": i} for i in range(num_cases)]

        first_compile_error_index = -1
        
        # Determine concurrency limit
        max_workers = min(os.cpu_count() // 2 , len(inputs)) if local_run else max(32, os.cpu_count() * 5)
        
        # Use asyncio.Semaphore to limit concurrency, mimicking ThreadPoolExecutor behavior
        sem = asyncio.Semaphore(max_workers)

        async def sem_process_single_case(i, *args):
            async with sem:
                try:
                    return await self._process_single_case(i, *args)
                except Exception as e:
                    logger.error(f"Test case {i} generated an exception: {e}")
                    traceback.print_exc()
                    return -1, {
                        "case_index": i,
                        "input": str(args[0]),
                        "expected_output": str(args[1]),
                        "api_request_error": f"Internal execution error: {e}",
                        "status": "internal_error",
                    }

        # Create tasks for all inputs
        tasks = []
        for i, stdin_data in enumerate(inputs):
            tasks.append(
                sem_process_single_case(
                    i,
                    stdin_data,
                    expected_outputs[i],
                    sandbox_fusion_url,
                    generation,
                    timeout,
                    memory_limit_mb,
                    language,
                    local_run,
                    concurrent_semaphore,
                    fn_name,
                )
            )

        # Use asyncio.as_completed to process results as they finish
        for future in asyncio.as_completed(tasks):
            result_status, metadata = await future
            index = metadata["case_index"]
            
            results[index] = result_status
            metadata_list[index] = metadata


            if result_status == -4:
                if first_compile_error_index == -1 or index < first_compile_error_index:
                    first_compile_error_index = index    
        # Post-processing for compile errors
        if first_compile_error_index != -1:
            logger.warning(
                f"Compile error detected in case {first_compile_error_index}. Marking subsequent cases as compile errors."
            )
            for i in range(first_compile_error_index + 1, num_cases):
                # Only update if not already processed (though it should be None or have a result)
                if results[i] != -4:  # Avoid overwriting if it somehow already got -4
                    results[i] = -4
                    # Update or create metadata for skipped cases due to compile error
                    if metadata_list[i] is None:  # If future failed before returning metadata
                        metadata_list[i] = {
                            "case_index": i,
                            "input": str(inputs[i]),
                            "expected_output": str(expected_outputs[i]),
                            "api_request_error": None,
                            "status": "compile_error_skipped",  # Indicate skipped due to prior compile error
                        }
                    else:  # If future completed but result is overridden
                        metadata_list[i]["status"] = "compile_error_skipped"

        logger.info(f"Correctness check finished. Results: {results}")
        breakpoint()
        return results, metadata_list

    async def execute_code(
        self,
        sandbox_fusion_url,
        memory_limit_mb,
        code, timeout=30, language="python", 
        ground_truth=None, local_run=False
    ):
        # Handle None or empty ground_truth
        if ground_truth is None:
            ground_truth = {}
            
        if "functional" in ground_truth:
            code = code + "\n" + ground_truth["functional"]
            result_status, metadata = await self._process_single_case(
                0, None, None, sandbox_fusion_url, code, timeout, memory_limit_mb, language, local_run
            )
            breakpoint()
            if metadata["run_status"] == "Finished":
                actual_output = metadata["stdout"] + metadata["stderr"]
                code_status = metadata["api_status"]
                logger.debug(f"actual_output from sandbox fusion: {actual_output}")
                return actual_output, code_status, metadata
            else:
                return "no stdout here", "Not Finished", metadata
        elif "inputs" in ground_truth and "outputs" in ground_truth:
            result_status, metadata = await self.check_correctness(
                sandbox_fusion_url, ground_truth,  code, timeout, memory_limit_mb, language, local_run
            )
            total_cases = len(result_status)
            if total_cases == 0:
                return "No test cases found.", "Success", {"status": "Success", "run_status": "Finished", "stdout": "Test cases pass rate:**0.00**\n No test cases found.", "stderr": "", "results": [], }
            breakpoint()
            passed_count = 0
            first_failure_meta = None
            final_code_status = "Success" # Assume success unless we find a failure
            case_duration = []

            for i, (status, meta) in enumerate(zip(result_status, metadata)):
                case_duration.append(meta['duration'])
                if status is True:
                    passed_count += 1
                else:
                    if first_failure_meta is None:
                        first_failure_meta = meta
                        final_code_status = meta.get("api_status", "Failed")
            
            final_metadata = {}
            pass_fail_list = [1 if status is True else 0 for status in result_status]
            final_metadata["pass_fail_list"] = pass_fail_list
            if first_failure_meta is None:
                stdout_str = f"Test cases pass rate: **1.00**\nAll {total_cases} test cases passed."
                stderr_str = ""
                final_metadata = {
                    "run_status": "Finished",
                    "api_status": "Success",
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                    "exit_code": 0,
                    "status": "success",
                    "pass_fail_list": pass_fail_list
                }
            else:
                pass_rate = passed_count / total_cases
                
                failed_input = first_failure_meta.get("input")
                actual_output_val = first_failure_meta.get("stdout")
                expected_output_val = first_failure_meta.get("expected_output")
                
                stdout_str = (
                    f"Test cases pass rate: **{pass_rate:.2f}**\n"
                    f"#Failed Test Case:\n"
                    f"  - Input: {failed_input}\n"
                    f"  - Your return value: {repr(actual_output_val)}\n"
                    f"  - Expected answer:  {repr(expected_output_val)}"
                )

                stderr_str = first_failure_meta.get("stderr", "")

                final_metadata = {
                    "run_status": first_failure_meta.get("run_status", "Finished"),
                    "api_status": first_failure_meta.get("api_status", "Failed"),
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                    "exit_code": first_failure_meta.get("exit_code", 1),
                    "status": first_failure_meta.get("status", "wrong_answer"),
                    "failed_case_index": first_failure_meta.get("case_index")
                }

            final_actual_output = final_metadata["stdout"]
            final_metadata["results"] = result_status
            if final_metadata["stderr"]:
                final_actual_output += "\n#Error Log:\n" + final_metadata["stderr"]
            final_metadata['duration'] = case_duration
                
            logger.debug(f"Aggregated actual_output: {final_actual_output}")
            return final_actual_output, final_code_status, final_metadata
        else:
            # Fallback: no ground_truth or unrecognized format, just execute the code
            result_status, metadata = await self._process_single_case(
                0, None, None, sandbox_fusion_url, code, timeout, memory_limit_mb, language, local_run
            )
            if metadata["run_status"] == "Finished":
                actual_output = metadata["stdout"] + metadata["stderr"]
                code_status = metadata["api_status"]
                logger.debug(f"actual_output from sandbox (no ground_truth): {actual_output}")
                return actual_output, code_status, metadata
            else:
                return "Execution did not finish", "Not Finished", metadata


class ToolRegistry:
    """Tool registry, manages available tools and their execution"""

    def __init__(self):
        self.tools = {}
        self.python_sandbox = PythonSandbox(
            timeout=TOOL_CONFIGS["python_timeout"], memory_limit=TOOL_CONFIGS["python_memory_limit"]
        )
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default tools in the registry"""
        # Python code interpreter
        self.register_tool(
            "code_interpreter",
            {
                "type": "function",
                "function": {
                    "name": "code_interpreter",
                    "description": "A tool for executing Python code in a safe sandbox environment.",
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string", "description": "The Python code to execute"}},
                        "required": ["code"],
                    },
                },
            },
        )

    def register_tool(self, name: str, tool_spec: dict[str, Any]):
        """Register a new tool in the registry"""
        self.tools[name] = tool_spec

    def get_tool_specs(self) -> list[dict[str, Any]]:
        """Get all tool specifications as a list"""
        return list(self.tools.values())

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool call with the given arguments"""
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"

        async with SEMAPHORE:
            if tool_name == "code_interpreter":
                return await self._execute_python(arguments)
            else:
                return f"Error: Tool '{tool_name}' not implemented"

    async def _execute_python(self, arguments: dict[str, Any]) -> str:
        """Execute Python code using the sandbox"""
        code = arguments.get("code", "")
        if not code.strip():
            return "Error: No code provided"

        # Execute code in sandbox
        result = await self.python_sandbox.execute_code(
            sandbox_fusion_url=self.sandbox_url,
            memory_limit_mb=4096, # Pass default memory limit
            code=code,
            timeout=TOOL_CONFIGS["python_timeout"],
            ground_truth=None # No ground truth for direct interpreter execution
        )
        # Extract the output string from the tuple result (output, status, metadata)
        if isinstance(result, tuple) and len(result) >= 1:
            return str(result[0])
        return str(result)


# Global tool registry instance
tool_registry = ToolRegistry()
