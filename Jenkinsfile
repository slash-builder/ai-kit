// Centralized, pinned Rust toolchain image — bump here to move every stage at once.
// Build, test, and publish run inside this container so the Rust version stays
// decoupled from the Jenkins agent host. Pinned to latest stable as of 2026-06.
def RUST_IMAGE = 'rust:1.94'

// Run `body` on an agent with `label`, as an optional leg (see the comment above the legs).
//   - The leg fails, or no agent picks it up, or it runs too long: mark the build UNSTABLE, do not fail it.
//   - An offline agent is the common case (the Macs are moved around). `nodesByLabel` is not available on
//     our controller, so allocation is bounded by an inactivity timeout instead: the build logs steadily
//     while a leg runs, so 6 minutes of silence means nothing is running and the leg is abandoned.
//     The outer timeout is the hard cap on a leg that does run.
// Never use this for the linux legs: those are the gate.
def optionalLeg(String name, String label, int minutes, Closure body) {
  try {
    timeout(time: minutes, unit: 'MINUTES') {
      timeout(time: 6, unit: 'MINUTES', activity: true) {
        node(label) {
          checkout scm
          body()
        }
      }
    }
  } catch (err) {
    unstable("${name}: ${err.getMessage()}")
  }
}

pipeline {
  // Pinned (not `any`): Build & Test and Publish both run a docker { reuseNode true }
  // sub-agent, which reuses whatever node this top-level agent lands on. `agent any`
  // let that land on any idle executor, including win32/saturn — hosts with no docker
  // binary — which fails immediately with "docker: command not found" (reproduced on
  // saturn in build #2, 2026-08-30). Pin to linux-build so the container-based legs are
  // deterministic.
  agent { label 'linux-build' }

  environment {
    NEXUS_URL  = 'https://nexus.softsurve.com'
    // RELATIVE on purpose -- do not restore `"${WORKSPACE}/.cargo"`.
    // That interpolates ONCE against the top-level agent's workspace, but
    // docker stages bind-mount only their own workspace, and under
    // concurrency Jenkins allocates `<job>@2`. The baked path then names a
    // directory not mounted in the container, and cargo dies with
    // "Read-only file system (os error 30)". Root-caused on message-kit
    // PR-14 builds 3/4, 2026-09-12. Cargo resolves a relative CARGO_HOME
    // against cwd, and no step here uses `cd`. (CI hardening 2026-09-13)
    CARGO_HOME = '.cargo'
  }

  stages {

    stage('Pre-flight') {
      steps {
        script {
          def msg = sh(script: 'git log -1 --pretty=%B', returnStdout: true).trim()
          if (msg.contains('[skip ci]')) {
            currentBuild.result = 'NOT_BUILT'
            error('Commit contains [skip ci] — aborting.')
          }
        }
      }
    }

    stage('Build & Test') {
      parallel {

        stage('linux-amd64') {
          // Unchanged from the pre-matrix pipeline: pinned rust:1.94 container,
          // reuses the linux-build node this pipeline is now pinned to.
          agent { docker { image RUST_IMAGE; reuseNode true } }
          steps {
            sh '''
              rustup component add rustfmt clippy
              cargo fmt --check
              cargo clippy --all-targets -- -D warnings
              cargo test
              cargo build --release
            '''
          }
        }

        stage('linux-arm64 (cross-compile)') {
          // No native Linux arm64 agent exists yet. AI Kit has no ML-framework
          // native deps wired in today — Candle/TFLite/ONNX are all still
          // commented out in Cargo.toml (see Cargo.toml lines ~55-57) — so the
          // dependency graph is pure Rust + serde/tokio/blake3/memmap2, which
          // cross-compiles cleanly from the existing linux-build agent. Revisit
          // this leg (native arm64 agent, or drop cross-compilation) once a real
          // inference backend with native/GPU deps lands; that may no longer
          // cross-compile cleanly.
          //
          // Build-only: a cross-compiled aarch64 binary can't execute on this
          // amd64 host without an emulator (no qemu-user/binfmt assumed here),
          // so `cargo test` is not run for this leg.
          // `-u root` is required: this stage apt-get installs the aarch64
          // cross-toolchain, and without it apt fails with
          //   E: Could not open lock file /var/lib/apt/lists/lock (13: Permission denied)
          // The build then continues and dies later at the confusing
          // "failed to find tool aarch64-linux-gnu-gcc" -- the apt failure is
          // non-fatal, so the real cause is 200 lines earlier. `reuseNode` is
          // kept so this stage does not queue for a second executor.
          agent { docker { image RUST_IMAGE; args '-u root'; reuseNode true } }
          steps {
            sh '''
              rustup target add aarch64-unknown-linux-gnu
              # Both packages are required: gcc-aarch64-linux-gnu alone is not enough —
              # blake3's C NEON implementation (blake3_neon.c) needs the aarch64 cross
              # sysroot headers too, or cc-rs fails with "bits/wordsize.h: No such file".
              # Verified locally (rust:1.94) 2026-08-29: fails with only gcc-aarch64-linux-gnu,
              # succeeds with libc6-dev-arm64-cross added.
              apt-get update -qq && apt-get install -y -qq --no-install-recommends \
                gcc-aarch64-linux-gnu libc6-dev-arm64-cross
              # Isolate this leg's build directory. The stages in this `parallel`
              # block share ONE workspace (`reuseNode true`), and this leg runs as
              # root so it can apt-get. Sharing `target/` means it writes
              # root-owned files that a concurrently-running sibling stage then
              # cannot touch:
              #   error: failed to open: .../target/release/.cargo-lock
              #   Permission denied (os error 13)
              # A chown at the end of this stage does NOT fix that -- the sibling
              # hits the file while this stage is still running. Separate target
              # dirs remove the contention instead of racing it.
              # (CI hardening 2026-09-18)
              export CARGO_TARGET_DIR="$PWD/target-arm64"
              export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
              cargo build --release --target aarch64-unknown-linux-gnu

              # Hand the workspace back. This stage runs as root (it has to,
              # to apt-get) and `reuseNode true` means it shares the SAME
              # workspace as every other stage, which runs as the Jenkins
              # uid. Without this, cargo artifacts written here stay
              # root-owned and the next stage dies with
              #   error: failed to open: .../target/release/.cargo-lock
              #   Permission denied (os error 13)
              # `--reference` copies the workspace root's own ownership
              # rather than hardcoding a uid. (CI hardening 2026-09-18)
              chown -R --reference="$PWD" "$PWD"
            '''
          }
        }

        // The Windows and macOS legs run on agents this repo does not control (mars, saturn), and
        // they come and go. They are verification, not a release gate, so they must never block
        // Publish. Two things used to:
        //   1. catchError defaults to buildResult FAILURE, and a declarative pipeline does not run
        //      later stages once the build is FAILURE -- so one failing leg skipped Publish.
        //   2. A stage-level `agent { label 'saturn' }` waits for an agent with no limit, so an
        //      offline saturn hung the whole build (and the PR check) until someone cancelled it.
        // optionalLeg() fixes both: a leg that cannot get an agent is abandoned and the build goes
        // UNSTABLE (yellow, not red); a leg that runs and fails does the same. A leg that takes too
        // long is stopped. Publish still runs on main, and the unstable result stays visible.
        stage('windows-amd64') {
          steps {
            script {
              optionalLeg('windows-amd64', 'win32', 40) {
                bat '''
                  rustup component add rustfmt clippy
                  cargo fmt --check || exit /b 1
                  cargo clippy --all-targets -- -D warnings || exit /b 1
                  cargo test || exit /b 1
                  cargo build --release || exit /b 1
                '''
              }
            }
          }
        }

        stage('macos-arm64') {
          steps {
            script {
              optionalLeg('macos-arm64', 'saturn', 40) {
                sh '''
                  rustup component add rustfmt clippy
                  cargo fmt --check
                  cargo clippy --all-targets -- -D warnings
                  cargo test
                  cargo build --release
                '''
              }
            }
          }
        }

      }
    }

    stage('Publish') {
      when { branch 'main' }
      agent { docker { image RUST_IMAGE; reuseNode true } }
      steps {
        withCredentials([
          usernamePassword(
            credentialsId: 'nexus-credentials',
            usernameVariable: 'NEXUS_USER',
            passwordVariable: 'NEXUS_PASS'
          ),
          usernamePassword(
            credentialsId: 'github-token',
            usernameVariable: 'GIT_USER',
            passwordVariable: 'GIT_TOKEN'
          )
        ]) {
          sh '''
            git config user.email "ci@softsurve.com"
            git config user.name  "Sol CI"

            set -eu

            origin=$(git config --get remote.origin.url)
            path=$(printf '%s' "$origin" | sed -E 's#(git@github.com:|https://github.com/)##; s#[.]git$##')
            remote="https://${GIT_USER}:${GIT_TOKEN}@github.com/${path}.git"

            # Tags are the version source of truth — bump the patch above the latest vX.Y.Z.
            git fetch --tags --quiet "$remote" || true
            latest=$(git tag -l 'v*.*.*' | sort -V | tail -1)
            if [ -z "$latest" ]; then
              next="0.1.0"
            else
              v=${latest#v}; maj=${v%%.*}; rest=${v#*.}; min=${rest%%.*}; pat=${rest##*.}
              next="${maj}.${min}.$((pat + 1))"
            fi
            echo "Publishing ${path} v${next}"

            # Set the crate version for this publish (ephemeral; tags stay the source of truth).
            if [ -f Cargo.toml ]; then
              sed -i 's/^version = ".*"/version = "'"$next"'"/' Cargo.toml
            fi

            # Cargo registry auth.
            mkdir -p "$CARGO_HOME"
            cat >> "$CARGO_HOME/config.toml" <<EOF
[registries.lockamy]
index = "sparse+${NEXUS_URL}/repository/cargo-group/"

[registries.lockamy-hosted]
index = "sparse+${NEXUS_URL}/repository/cargo-hosted/"

[registry]
default = "lockamy"
EOF
            # Nexus speaks HTTP Basic; cargo sends the token verbatim as the Authorization header,
            # so the token must be "Basic <base64(user:pass)>" — not a bare user:pass.
            BASIC="Basic $(printf '%s:%s' "${NEXUS_USER}" "${NEXUS_PASS}" | base64 -w0)"
            printf '[registries.lockamy]\ntoken = "%s"\n[registries.lockamy-hosted]\ntoken = "%s"\n' \
              "${BASIC}" "${BASIC}" >> "$CARGO_HOME/credentials.toml"
            chmod 0600 "$CARGO_HOME/credentials.toml"

            # Publish (allow-dirty: the version was just set in-tree).
            # Idempotent: a version already present on Nexus is treated as success,
            # so re-running a build that already published doesn't fail the pipeline.
            set +e
            pub_out=$(cargo publish --registry lockamy-hosted --allow-dirty 2>&1)   # resolve via group (default), publish to hosted
            pub_rc=$?
            set -e
            printf '%s\n' "$pub_out"
            if [ "$pub_rc" -ne 0 ]; then
              if printf '%s' "$pub_out" | grep -qiE 'already (exists|uploaded)|already been uploaded|version .* is already'; then
                echo "v${next} already published — treating as success (idempotent)."
              else
                exit "$pub_rc"
              fi
            fi

            # Record the release as a tag and push it back to origin (skip if it already exists).
            if git rev-parse -q --verify "refs/tags/v${next}" >/dev/null; then
              echo "Tag v${next} already exists locally — skipping tag."
            else
              git tag "v${next}"
              git push "$remote" "v${next}" || echo "Tag push skipped (already on origin)."
            fi
          '''
        }
      }
    }

  }
  post {
    success { echo 'ai-kit pipeline succeeded' }
    failure { echo 'Pipeline failed.' }
    always  { cleanWs() }
  }
}
