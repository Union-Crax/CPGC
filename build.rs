fn main() {
    // When built with --features pretrained, verify the weight file exists
    // so the user gets a clear error instead of a cryptic include_bytes! failure.
    #[cfg(feature = "pretrained")]
    {
        let path = std::path::Path::new("src/predictor/pretrained.bin");
        if !path.exists() {
            panic!(
                "\n\nERROR: --features pretrained requires src/predictor/pretrained.bin\n\
                 Run experiments/pretrain_colab.ipynb on Google Colab, download the\n\
                 exported cpgc_pretrained.bin, and place it at src/predictor/pretrained.bin\n"
            );
        }
        // Re-run if the weights change
        println!("cargo:rerun-if-changed=src/predictor/pretrained.bin");
    }
    println!("cargo:rerun-if-changed=build.rs");
}
