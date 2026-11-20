.PHONY: check test clippy fmt doc release

check:
	cargo check --workspace --all-targets

test:
	cargo test --workspace

clippy:
	cargo clippy --workspace --all-targets -- -D warnings

fmt:
	cargo fmt --all

doc:
	cargo doc --workspace --no-deps

release:
	cargo build --release --bin tufazip
