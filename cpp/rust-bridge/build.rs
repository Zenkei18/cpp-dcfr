fn main() {
    cxx_build::bridge("src/lib.rs")
        .compile("pokers-bridge");
        
    println!("cargo:rerun-if-changed=src/lib.rs");
}
