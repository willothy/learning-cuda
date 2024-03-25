build_dir := invocation_directory() / "build"

[private]
list:
    @just readme
    @just --list

# Use Glow to render the README file
readme:
    @glow README.md

# Build the specified dir
build subproject:
    @if [ -z {{ subproject }} ]; then echo "Please specify a subproject to build"; exit 1; fi
    @test -d {{ subproject }} || (echo "Subproject '{{ subproject }}' does not exist" && exit 1)
    @test -f {{ subproject }}/justfile || (echo "Subproject '{{ subproject }}' does not have a justfile" && exit 1)
    mkdir -p {{ build_dir }}
    @just --set build_dir {{ build_dir }} -d {{ subproject }} --justfile {{ subproject }}/justfile build

run subproject:
    @if [ -z {{ subproject }} ]; then echo "Please specify a subproject to build"; exit 1; fi
    @test -d {{ subproject }} || (echo "Subproject '{{ subproject }}' does not exist" && exit 1)
    @test -f {{ subproject }}/justfile || (echo "Subproject '{{ subproject }}' does not have a justfile" && exit 1)
    mkdir -p {{ build_dir }}
    @just --set build_dir {{ build_dir }} -d {{ subproject }} --justfile {{ subproject }}/justfile run

clean:
    @rm -rf {{ build_dir }}
