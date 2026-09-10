# ==============================================================================
# tee_run_log.jl — Save the post-run console summary into the results folder
# ==============================================================================
#
# After ADMM converges / hits max_iter (or the social planner finishes), the
# save_* routines print cost metrics and the run summary. Wrap those prints
# with with_run_summary_log so the same text is written to run_summary.txt.
#
# ==============================================================================

if !isdefined(@__MODULE__, :with_run_summary_log)

using Dates
using Logging

struct TeeStream <: IO
    primary::IO
    log::IO
end

function Base.unsafe_write(s::TeeStream, p::Ptr{UInt8}, n::UInt)
    unsafe_write(s.primary, p, n)
    nlog = unsafe_write(s.log, p, n)
    flush(s.log)
    return nlog
end

Base.write(s::TeeStream, x::UInt8) = (write(s.primary, x); write(s.log, x); flush(s.log); 1)
Base.flush(s::TeeStream) = (flush(s.primary); flush(s.log); nothing)
Base.isopen(s::TeeStream) = isopen(s.primary) && isopen(s.log)
Base.iswritable(s::TeeStream) = iswritable(s.primary)
Base.isreadable(::TeeStream) = false

"""Run `f()` on the terminal and copy that output to `<results_dir>/run_summary.txt`."""
function with_run_summary_log(f::Function, results_dir::AbstractString;
                              filename::String = "run_summary.txt")
    isdir(results_dir) || mkpath(results_dir)
    log_path = joinpath(results_dir, filename)
    open(log_path, "w") do log
        script = !isempty(PROGRAM_FILE) ? PROGRAM_FILE : "repl"
        println(log, "# run_summary.txt")
        println(log, "# script:  ", script)
        println(log, "# written: ", Dates.now())
        println(log, "# results: ", results_dir)
        println(log)
        flush(log)

        orig_out = stdout
        orig_err = stderr
        orig_logger = global_logger()
        tee_out = TeeStream(orig_out, log)
        tee_err = TeeStream(orig_err, log)
        redirect_stdout(tee_out)
        redirect_stderr(tee_err)
        global_logger(ConsoleLogger(tee_err))
        try
            f()
        finally
            global_logger(orig_logger)
            redirect_stdout(orig_out)
            redirect_stderr(orig_err)
            flush(log)
        end
    end
    return log_path
end

end # !isdefined with_run_summary_log
