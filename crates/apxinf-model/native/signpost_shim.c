// os_signpost operator-timeline shim for the Clef request path.
//
// os_signpost's public interval API is a set of macros over clang's
// compile-time __builtin_os_log_format: the format string and every argument
// must be visible to the compiler, so the macros cannot be expanded from Rust.
// This translation unit switches on the operator kind and calls each macro
// with a literal format, then exposes two uniform entry points:
//
//   apx_sp_log()                          the Points of Interest log handle
//   apx_sp_begin(kind, log, spid, ...)    variadic; reads exactly the
//                                          arguments Rust passes for `kind`
//   apx_sp_end(kind, log, spid)           matching interval end, empty format
//
// The category is OS_LOG_CATEGORY_POINTS_OF_INTEREST so Instruments' Points
// of Interest instrument records the intervals alongside Metal System Trace's
// command-buffer lane. The kind values must match the constants in
// crates/apxinf-model/src/metal_clef/signpost.rs.
#include <os/signpost.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

os_log_t apx_sp_log(void) {
    // Created once per process; Rust only calls this behind a OnceLock.
    static os_log_t log;
    if (log == NULL) {
        log = os_log_create("com.apxinf.clef", OS_LOG_CATEGORY_POINTS_OF_INTEREST);
    }
    return log;
}

enum {
    APX_SP_REQUEST = 0,
    APX_SP_PREPROCESS = 1,
    APX_SP_ENCODE = 2,
    APX_SP_VISION = 3,
    APX_SP_LAYER = 4,
    APX_SP_HEAD = 5,
    APX_SP_ANE_WAIT = 6,
    APX_SP_ANE_PREDICT = 7,
    APX_SP_SYNC = 8,
};

void apx_sp_begin(unsigned int kind, os_log_t log, uint64_t spid, ...) {
    va_list ap;
    va_start(ap, spid);
    switch (kind) {
    case APX_SP_REQUEST: {
        size_t images = va_arg(ap, size_t);
        os_signpost_interval_begin(log, spid, "request",
                                   "images=%{public}zu", images);
        break;
    }
    case APX_SP_PREPROCESS: {
        size_t images = va_arg(ap, size_t);
        os_signpost_interval_begin(log, spid, "preprocess",
                                   "images=%{public}zu", images);
        break;
    }
    case APX_SP_ENCODE: {
        // The token count is unknown until `encode` returns (the prompt is
        // built inside it), so the interval carries the request shape.
        size_t images = va_arg(ap, size_t);
        size_t questions = va_arg(ap, size_t);
        os_signpost_interval_begin(log, spid, "encode",
            "images=%{public}zu questions=%{public}zu", images, questions);
        break;
    }
    case APX_SP_VISION: {
        size_t rows = va_arg(ap, size_t);
        os_signpost_interval_begin(log, spid, "vision",
                                   "rows=%{public}zu", rows);
        break;
    }
    case APX_SP_LAYER: {
        size_t index = va_arg(ap, size_t);
        const char *mixer = va_arg(ap, const char *);
        size_t tokens = va_arg(ap, size_t);
        int cached = va_arg(ap, int);
        os_signpost_interval_begin(log, spid, "layer",
            "index=%{public}zu mixer=%{public}s tokens=%{public}zu cached=%{public}d",
            index, mixer, tokens, cached);
        break;
    }
    case APX_SP_HEAD: {
        size_t tokens = va_arg(ap, size_t);
        size_t questions = va_arg(ap, size_t);
        os_signpost_interval_begin(log, spid, "head",
            "tokens=%{public}zu questions=%{public}zu", tokens, questions);
        break;
    }
    case APX_SP_ANE_WAIT: {
        size_t layer = va_arg(ap, size_t);
        size_t tokens = va_arg(ap, size_t);
        os_signpost_interval_begin(log, spid, "ane_wait",
            "layer=%{public}zu tokens=%{public}zu", layer, tokens);
        break;
    }
    case APX_SP_ANE_PREDICT: {
        size_t layer = va_arg(ap, size_t);
        size_t tiles = va_arg(ap, size_t);
        size_t tokens = va_arg(ap, size_t);
        os_signpost_interval_begin(log, spid, "ane_predict",
            "layer=%{public}zu tiles=%{public}zu tokens=%{public}zu",
            layer, tiles, tokens);
        break;
    }
    case APX_SP_SYNC: {
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

void apx_sp_end(unsigned int kind, os_log_t log, uint64_t spid) {
    switch (kind) {
    case APX_SP_REQUEST:
        os_signpost_interval_end(log, spid, "request");
        break;
    case APX_SP_PREPROCESS:
        os_signpost_interval_end(log, spid, "preprocess");
        break;
    case APX_SP_ENCODE:
        os_signpost_interval_end(log, spid, "encode");
        break;
    case APX_SP_VISION:
        os_signpost_interval_end(log, spid, "vision");
        break;
    case APX_SP_LAYER:
        os_signpost_interval_end(log, spid, "layer");
        break;
    case APX_SP_HEAD:
        os_signpost_interval_end(log, spid, "head");
        break;
    case APX_SP_ANE_WAIT:
        os_signpost_interval_end(log, spid, "ane_wait");
        break;
    case APX_SP_ANE_PREDICT:
        os_signpost_interval_end(log, spid, "ane_predict");
        break;
    case APX_SP_SYNC:
        os_signpost_interval_end(log, spid, "sync");
        break;
    default:
        break;
    }
}
