fn main() {
    // Link Apple Accelerate framework for BLAS (cblas_dgemm) on macOS.
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
