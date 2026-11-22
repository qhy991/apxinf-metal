//! Standalone IOSurface feasibility probe for Issue #161 P2 (gate I arm,
//! doc/ane-prefill-offload.md §1.1). Zero crates.io dependencies: the
//! IOSurface framework is loaded with `dlopen`/`dlsym`, so there is no `link`
//! attribute and the binary never needs the framework at build time.
//!
//! Probe shape (frozen in doc §1.1): `[1, 4, 8192, 128]` BF16 = 98,552,320 B.
//! `cargo run --release -p apxinf-model --example iosurface_probe -- [--out <json>]`.
//! One JSON line on stdout; `--out` additionally writes it to a path.
#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

#[cfg(target_os = "macos")]
mod native {
    use serde_json::{json, Value};
    use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString, OsStr};
    use std::os::unix::ffi::OsStrExt;
    use std::path::PathBuf;
    use std::time::Instant;

    /// Doc §1.1: KV slab handed to the ANE arm. 4 kv_heads x 8192 positions
    /// x 128 head_dim, BF16. The allocator-level size is what the gate
    /// compares against (row padding and hardware alignment may add bytes).
    pub const IOSURFACE_BYTES: usize = 98_552_320;
    const WIDTH: c_uint = 1;
    const HEIGHT: c_uint = 4;
    const BYTES_PER_ROW: c_uint = 8192 * 128 * 2;

    type CFTypeRef = *const c_void;
    type CFDictionaryRef = *const c_void;
    type CFMutableDictionaryRef = *mut c_void;
    type CFNumberRef = *const c_void;
    type CFAllocatorRef = *const c_void;
    type CFIndex = isize;
    type CFNumberType = i32;
    type IOSurfaceRef = *const c_void;
    const KCF_NUMBER_SINT32_TYPE: CFNumberType = 3;
    /// `kCFAllocatorDefault` is `nil`, ABI-correct on macOS.
    const K_CF_ALLOCATOR_DEFAULT: CFAllocatorRef = std::ptr::null();

    type FnCreate = unsafe extern "C" fn(CFDictionaryRef) -> IOSurfaceRef;
    type FnAllocSize = unsafe extern "C" fn(IOSurfaceRef) -> usize;
    type FnGetU32 = unsafe extern "C" fn(IOSurfaceRef) -> c_uint;
    type FnCFRelease = unsafe extern "C" fn(CFTypeRef);
    type FnCFNumberCreate =
        unsafe extern "C" fn(CFAllocatorRef, CFNumberType, *const c_void) -> CFNumberRef;
    type FnCFDictCreateMutable = unsafe extern "C" fn(
        CFAllocatorRef,
        CFIndex,
        *const c_void,
        *const c_void,
    ) -> CFMutableDictionaryRef;
    type FnCFDictSetValue =
        unsafe extern "C" fn(CFMutableDictionaryRef, *const c_void, *const c_void);

    #[link(name = "System")]
    extern "C" {
        fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> c_int;
        fn dlerror() -> *const c_char;
    }
    const RTLD_LAZY: c_int = 1;
    const RTLD_NOW: c_int = 2;

    /// A dynamically-resolved symbol.
    struct Symbol(*mut c_void);
    impl Symbol {
        unsafe fn get(handle: *mut c_void, name: &[u8]) -> Result<Self, String> {
            let cname = CString::new(name).map_err(|e| format!("cstr {name:?}: {e}"))?;
            let p = unsafe { dlsym(handle, cname.as_ptr()) };
            if p.is_null() {
                let err = unsafe { dlerror() };
                let msg = if err.is_null() {
                    "no detail".to_string()
                } else {
                    unsafe { CStr::from_ptr(err).to_string_lossy().into_owned() }
                };
                Err(format!("dlsym {}: {msg}", cname.to_string_lossy()))
            } else {
                Ok(Self(p))
            }
        }
        /// Resolve this symbol to a function pointer type. The pointer is
        /// only valid while the owning `Library` stays mapped (callers must
        /// hold the library until they finish using it).
        unsafe fn as_fn<T: Copy>(&self) -> T {
            unsafe { std::mem::transmute_copy(&self.0) }
        }
        /// CFString constants (kIOSurfaceWidth, ...) are exported as global
        /// variables whose *contents* are the CFStringRef; dlsym returns the
        /// address of the variable, so read it once to get the actual key.
        unsafe fn as_cfstring(&self) -> CFTypeRef {
            unsafe { *(self.0 as *const CFTypeRef) }
        }
    }

    /// A `dlopen`ed library. Held until its objects are released: unloading
    /// IOSurface.framework while an IOSurface CF object is still live would
    /// leave the object's class vtable pointing into unmapped memory.
    struct Library(*mut c_void);
    impl Library {
        fn open(path: &OsStr, flags: c_int) -> Result<Self, String> {
            let cpath = CString::new(path.as_bytes()).map_err(|e| format!("cstr: {e}"))?;
            let handle = unsafe { dlopen(cpath.as_ptr(), flags) };
            if handle.is_null() {
                let err = unsafe { dlerror() };
                let msg = if err.is_null() {
                    "no detail".to_string()
                } else {
                    unsafe { CStr::from_ptr(err).to_string_lossy().into_owned() }
                };
                Err(format!("dlopen {}: {msg}", cpath.to_string_lossy()))
            } else {
                Ok(Self(handle))
            }
        }
        unsafe fn symbol(&self, name: &[u8]) -> Result<Symbol, String> {
            unsafe { Symbol::get(self.0, name) }
        }
        /// Resolve a symbol straight to a function pointer. One unsafe
        /// call site per symbol instead of `symbol()` + `as_fn()`.
        unsafe fn fn_of<T: Copy>(&self, name: &[u8]) -> Result<T, String> {
            let symbol = unsafe { self.symbol(name)? };
            Ok(unsafe { symbol.as_fn::<T>() })
        }
    }
    impl Drop for Library {
        fn drop(&mut self) {
            unsafe { dlclose(self.0) };
        }
    }

    fn probe_iosurface() -> Value {
        const FRAMEWORK: &str = "/System/Library/Frameworks/IOSurface.framework/IOSurface";
        const CF_FRAMEWORK: &str =
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation";

        let probe = (|| -> Result<Value, String> {
            let io = Library::open(OsStr::new(FRAMEWORK), RTLD_NOW)?;
            let cf = Library::open(OsStr::new(CF_FRAMEWORK), RTLD_LAZY)?;

            let create: FnCreate = unsafe { io.fn_of(b"IOSurfaceCreate")? };
            let alloc_size: FnAllocSize = unsafe { io.fn_of(b"IOSurfaceGetAllocSize")? };
            let get_width: FnGetU32 = unsafe { io.fn_of(b"IOSurfaceGetWidth")? };
            let get_height: FnGetU32 = unsafe { io.fn_of(b"IOSurfaceGetHeight")? };
            let get_bpr: FnGetU32 = unsafe { io.fn_of(b"IOSurfaceGetBytesPerRow")? };
            let get_plane_count: FnGetU32 = unsafe { io.fn_of(b"IOSurfaceGetPlaneCount")? };
            let cf_release: FnCFRelease = unsafe { cf.fn_of(b"CFRelease")? };
            let cf_number_create: FnCFNumberCreate = unsafe { cf.fn_of(b"CFNumberCreate")? };
            let cf_dict_create_mutable: FnCFDictCreateMutable =
                unsafe { cf.fn_of(b"CFDictionaryCreateMutable")? };
            let cf_dict_set_value: FnCFDictSetValue = unsafe { cf.fn_of(b"CFDictionarySetValue")? };

            // Build the property dictionary. Keys are symbols resolved from
            // the loaded IOSurface dylib and must stay mapped while the
            // surface exists.
            let dict = unsafe {
                cf_dict_create_mutable(
                    K_CF_ALLOCATOR_DEFAULT,
                    0,
                    std::ptr::null(),
                    std::ptr::null(),
                )
            };
            if dict.is_null() {
                return Err("CFDictionaryCreateMutable returned null".to_string());
            }
            // `keep` holds the resolved key symbols and CFNumber objects
            // alive for as long as `dict` exists; the closure below borrows
            // it mutably, so it is declared before `add_number`.
            let mut keep: Vec<(CFTypeRef, CFNumberRef)> = Vec::new();
            let mut add_number = |name: &[u8], value: c_uint| -> Result<(), String> {
                // kIOSurface* keys are exported by IOSurface, not CF.
                let key = unsafe { io.symbol(name)?.as_cfstring() };
                let num = unsafe {
                    cf_number_create(
                        K_CF_ALLOCATOR_DEFAULT,
                        KCF_NUMBER_SINT32_TYPE,
                        &value as *const c_uint as *const c_void,
                    )
                };
                if num.is_null() {
                    return Err(format!("CFNumberCreate({name:?}) returned null"));
                }
                keep.push((key, num));
                unsafe { cf_dict_set_value(dict, key, num as CFTypeRef) };
                Ok(())
            };
            add_number(b"kIOSurfaceWidth", WIDTH)?;
            add_number(b"kIOSurfaceHeight", HEIGHT)?;
            add_number(b"kIOSurfaceBytesPerRow", BYTES_PER_ROW)?;
            add_number(b"kIOSurfaceBytesPerElement", 2)?;

            let start = Instant::now();
            let surface = unsafe { create(dict as CFDictionaryRef) };
            let create_ms = start.elapsed().as_secs_f64() * 1e3;
            unsafe { cf_release(dict as CFTypeRef) };
            if surface.is_null() {
                return Err("IOSurfaceCreate returned null".to_string());
            }

            let (size, width, height, bpr, plane_count) = unsafe {
                (
                    alloc_size(surface),
                    get_width(surface),
                    get_height(surface),
                    get_bpr(surface),
                    get_plane_count(surface),
                )
            };
            unsafe { cf_release(surface as CFTypeRef) };
            // `io` and `cf` drop here, after the last release.

            Ok(json!({
                "probe": "iosurface",
                "framework": FRAMEWORK,
                "available": true,
                "requested_bytes": IOSURFACE_BYTES,
                "alloc_size": size,
                "alloc_size_delta": size as i64 - IOSURFACE_BYTES as i64,
                "width": width,
                "height": height,
                "bytes_per_row": bpr,
                "plane_count": plane_count,
                "create_ms": create_ms,
                "create_ms_median": create_ms,
            }))
        })();

        match probe {
            Ok(value) => value,
            Err(reason) => json!({
                "probe": "iosurface",
                "framework": FRAMEWORK,
                "available": false,
                "unavailable_reason": reason,
            }),
        }
    }

    pub fn run(out: Option<PathBuf>) -> Result<(), String> {
        let value = probe_iosurface();
        let text = serde_json::to_string(&value).map_err(|e| format!("serde_json: {e}"))?;
        if let Some(path) = out {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| format!("mkdir {parent:?}: {e}"))?;
            }
            std::fs::write(&path, format!("{text}\n"))
                .map_err(|e| format!("write {path:?}: {e}"))?;
        }
        println!("{text}");
        Ok(())
    }
}

use std::path::PathBuf;

fn main() {
    let mut out: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => {
                let value = args.next().unwrap_or_else(|| {
                    eprintln!("iosurface_probe: --out needs a value");
                    std::process::exit(1);
                });
                out = Some(PathBuf::from(value));
            }
            other => {
                eprintln!("iosurface_probe: unknown argument {other:?}");
                std::process::exit(1);
            }
        }
    }
    #[cfg(target_os = "macos")]
    if let Err(reason) = native::run(out) {
        eprintln!("iosurface_probe: {reason}");
        std::process::exit(1);
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = out;
        eprintln!("iosurface_probe requires macOS");
        std::process::exit(1);
    }
}
