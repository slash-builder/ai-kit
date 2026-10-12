//! The contract every [`CredentialSource`] must keep, as a reusable check. **No network, no dependencies.**
//!
//! The kit's own sources ([`DirCredentialSource`](super::DirCredentialSource), [`InMemoryCredentials`](super::InMemoryCredentials))
//! run it, and so should any source a host application writes (an OS credential store backed one in Courier, for example). A source that
//! passes has these properties, which is what lets an adapter treat "no credential" as "not configured" and never fall back:
//!
//! 1. **It returns what was stored**, trimmed of surrounding whitespace, and re-reads it on every call (a rotated secret is picked up
//!    without a restart).
//! 2. **It fails closed.** A missing credential is `Unconfigured`; a blank one is `Empty` (or `Unconfigured`); a name that is not a plain
//!    name (empty, `.`, `..`, containing a path separator, absolute) is `BadName` and the store is never asked for it. It never returns
//!    an empty secret and never panics.
//! 3. **No environment fallback.** An environment variable that happens to carry the credential's name (as written, upper-cased, or with
//!    `-` turned into `_`) is ignored.
//! 4. **Nothing leaks.** Neither the `Secret`'s `Debug` nor any error's `Display` or `Debug` contains the stored value. (A source's own
//!    `Debug` is not checked, because the trait does not require one; do not put the value in it.)
//!
//! Permission checks (the directory source insists on mode `0600` or stricter) are specific to that source and are tested there. A source
//! backed by an OS store replaces them with the store's own access control, and says so in its own documentation.
//!
//! ```ignore
//! struct MyFixture;
//! impl CredentialFixture for MyFixture {
//!     fn name(&self) -> &'static str { "os-store" }
//!     fn setup(&self, entries: &[(&str, &str)]) -> Setup { /* build the store holding exactly `entries` */ }
//! }
//! #[test] fn my_source_conforms() { ai_kit::credentials::assert_conforms(&MyFixture); }
//! ```

use super::{CredentialError, CredentialSource};
use std::any::Any;
use std::sync::atomic::{AtomicU32, Ordering};

/// A source holding exactly the entries asked for, plus a way to change one (rotation).
pub struct Setup {
    pub source: Box<dyn CredentialSource>,
    /// Store `value` under `name`, replacing what was there. The same source must see it on its next `get`.
    pub set: Box<dyn Fn(&str, &str) + Send + Sync>,
    /// Anything that must outlive the check (a temp directory, a test keychain).
    pub guard: Box<dyn Any + Send>,
}

pub trait CredentialFixture {
    fn name(&self) -> &'static str;
    /// Build a source that holds **exactly** `entries` (name, value) and nothing else. A value may be blank; a store that cannot hold a
    /// blank value should return [`CredentialFixture::supports_blank_values`] `false`.
    fn setup(&self, entries: &[(&str, &str)]) -> Setup;
    fn supports_blank_values(&self) -> bool {
        true
    }
}

/// A value unlikely to appear anywhere by accident, so "it leaked" is unambiguous.
const VALUE: &str = "conf-SECRET-7f3a9c1e-do-not-print";
static N: AtomicU32 = AtomicU32::new(0);

fn unique(prefix: &str) -> String {
    format!(
        "{prefix}-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst)
    )
}

fn assert_clean(who: &str, what: &str, text: &str) {
    assert!(
        !text.contains(VALUE) && !text.contains("7f3a9c1e"),
        "{who}: {what} contains the stored value: {text}"
    );
}

/// Run the whole contract against `fixture`. Panics with a message naming the fixture and the broken property.
pub fn assert_conforms(fixture: &dyn CredentialFixture) {
    let who = fixture.name();
    let present = unique("present");
    let padded = unique("padded");
    let blank = unique("blank");
    let missing = unique("missing");

    let mut entries: Vec<(&str, &str)> = vec![(&present, VALUE), (&padded, "  padded-value \n")];
    if fixture.supports_blank_values() {
        entries.push((&blank, "  \n"));
    }
    let s = fixture.setup(&entries);

    // 1. returns what was stored, trimmed
    let got = s
        .source
        .get(&present)
        .unwrap_or_else(|e| panic!("{who}: a stored credential was not returned: {e}"));
    assert_eq!(got.expose(), VALUE, "{who}: wrong value returned");
    assert_eq!(
        s.source.get(&padded).map(|v| v.expose().to_string()).ok(),
        Some("padded-value".to_string()),
        "{who}: surrounding whitespace must be trimmed"
    );

    // 2. fails closed
    match s.source.get(&missing) {
        Err(CredentialError::Unconfigured(n)) => {
            assert_eq!(n, missing, "{who}: the error should name the credential")
        }
        other => panic!("{who}: a missing credential must be Unconfigured, got {other:?}"),
    }
    if fixture.supports_blank_values() {
        assert!(
            matches!(
                s.source.get(&blank),
                Err(CredentialError::Empty(_)) | Err(CredentialError::Unconfigured(_))
            ),
            "{who}: a blank credential must be Empty or Unconfigured, never a usable secret"
        );
    }
    for bad in ["", ".", "..", "../x", "a/b", "/abs"] {
        match s.source.get(bad) {
            Err(CredentialError::BadName(_)) => {}
            other => panic!("{who}: name {bad:?} must be BadName, got {other:?}"),
        }
    }

    // 3. no environment fallback
    let vars = [
        missing.clone(),
        missing.to_uppercase(),
        missing.replace('-', "_").to_uppercase(),
    ];
    for v in &vars {
        std::env::set_var(v, VALUE);
    }
    let r = s.source.get(&missing);
    for v in &vars {
        std::env::remove_var(v);
    }
    assert!(
        matches!(r, Err(CredentialError::Unconfigured(_))),
        "{who}: an environment variable must never supply a credential, got {r:?}"
    );

    // re-read on every call: rotation is picked up without rebuilding the source
    (s.set)(&present, "rotated-value");
    assert_eq!(
        s.source.get(&present).map(|v| v.expose().to_string()).ok(),
        Some("rotated-value".to_string()),
        "{who}: a rotated credential must be seen on the next get"
    );
    (s.set)(&present, VALUE);

    // 4. nothing leaks
    let got = s.source.get(&present).unwrap();
    assert_clean(who, "the Secret's Debug", &format!("{got:?}"));
    let mut errors: Vec<CredentialError> = Vec::new();
    errors.extend(s.source.get(&missing).err());
    errors.extend(s.source.get("../x").err());
    if fixture.supports_blank_values() {
        errors.extend(s.source.get(&blank).err());
    }
    for e in &errors {
        assert_clean(who, "an error's Display", &e.to_string());
        assert_clean(who, "an error's Debug", &format!("{e:?}"));
    }
    drop(s.guard);
}
