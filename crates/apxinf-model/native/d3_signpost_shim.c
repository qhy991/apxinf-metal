// os_signpost operator-timeline shim for the d3-lite request path.
//
// os_signpost's public interval API is a set of macros over clang's
// compile-time __builtin_os_log_format: the format string and every argument
// must be visible to the compiler, so the macros cannot be expanded from Rust.
// This translation unit switches on the operator kind and calls each macro
// with a literal format, then exposes two uniform entry points:
//
//   apx_d3_sp_log()                        the Points of Interest log handle
//   apx_d3_sp_begin(kind, log, spid, ...)  variadic; reads exactly the
//                                           arguments Rust passes for `kind`
//   apx_d3_sp_end(kind, log, spid)         matching interval end, empty format
//
// The category is OS_LOG_CATEGORY_POINTS_OF_INTEREST so Instruments' Points
// of Interest instrument records the intervals alongside Metal System Trace's
// command-buffer lane. The kind values must match the constants in
// crates/apxinf-model/src/metal_d3_lite/signpost.rs. Symbol names are d3-prefixed
// so a build that also carries the Clef line's shim links without a clash.
#include <os/signpost.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

os_log_t apx_d3_sp_log(void) {
    // Created once per process; Rust only calls this behind a OnceLock.
    static os_log_t log;
    if (log == NULL) {
        log = os_log_create("com.apxinf.d3lite", OS_LOG_CATEGORY_POINTS_OF_INTEREST);
    }
    return log;
}

enum {
    APX_D3_SP_REQUEST = 0,
    APX_D3_SP_INPUT = 1,
    APX_D3_SP_VISION = 2,
    APX_D3_SP_FORWARD = 3,
    APX_D3_SP_SYNC = 4,
};

void apx_d3_sp_begin(unsigned int kind, os_log_t log, uint64_t spid, ...) {
    va_list ap;
    va_start(ap, spid);
    switch (kind) {
    case APX_D3_SP_REQUEST: {
        size_t tokens = va_arg(ap, size_t);
        size_t images = va_arg(ap, size_t);
        os_signpost_interval_begin(log, spid, "request",
            "tokens=%{public}zu images=%{public}zu", tokens, images);
        break;
    }
    case APX_D3_SP_INPUT: {
        size_t pixels = va_arg(ap, size_t);
        os_signpost_interval_begin(log, spid, "input",
                                   "pixels=%{public}zu", pixels);
        break;
    }
    case APX_D3_SP_VISION: {
        size_t image_tokens = va_arg(ap, size_t);
        os_signpost_interval_begin(log, spid, "vision",
                                   "image_tokens=%{public}zu", image_tokens);
        break;
    }
    case APX_D3_SP_FORWARD: {
        // The text callable has no image tokens; the image callable carries
        // both, so one literal format serves both.
        size_t tokens = va_arg(ap, size_t);
        size_t image_tokens = va_arg(ap, size_t);
        os_signpost_interval_begin(log, spid, "forward",
            "tokens=%{public}zu image_tokens=%{public}zu", tokens, image_tokens);
        break;
    }
    case APX_D3_SP_SYNC: {
        size_t logits = va_arg(ap, size_t);
        os_signpost_interval_begin(log, spid, "sync",
                                   "logits=%{public}zu", logits);
        break;
    }
    default:
        break;
    }
    va_end(ap);
}

void apx_d3_sp_end(unsigned int kind, os_log_t log, uint64_t spid) {
    switch (kind) {
    case APX_D3_SP_REQUEST:
        os_signpost_interval_end(log, spid, "request");
        break;
    case APX_D3_SP_INPUT:
        os_signpost_interval_end(log, spid, "input");
        break;
    case APX_D3_SP_VISION:
        os_signpost_interval_end(log, spid, "vision");
        break;
    case APX_D3_SP_FORWARD:
        os_signpost_interval_end(log, spid, "forward");
        break;
    case APX_D3_SP_SYNC:
        os_signpost_interval_end(log, spid, "sync");
        break;
    default:
        break;
    }
}
