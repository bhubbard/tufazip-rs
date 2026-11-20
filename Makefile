.PHONY: check test clippy fmt doc pages bench release

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

pages:
	RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
	rm -rf _site
	mkdir -p _site/api
	cp -R docs/. _site/
	cp -R target/doc/. _site/api/
	rm -f _site/api/.lock
	touch _site/.nojekyll

bench:
	cargo run --release --example benchmark -- --backend cpu

release:
	cargo build --release --bin tufazip
