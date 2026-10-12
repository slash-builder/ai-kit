//! Provider credentials: where a secret comes from, and how it is kept out of everything else. **No network.**
//!
//! A provider is configured with a *name* (`"anthropic-api-key"`), never a key. The production source reads
//! `$CREDENTIALS_DIRECTORY/<name>` (systemd `LoadCredential=`), and it **fails closed**: a missing, empty, unreadable or
//! group/world-accessible file means the provider is `Unconfigured` and is never constructed. There is **no fallback to environment
//! variables, argv or a config file** (hearth `docs/encryption.md` section 3.5, CLUS-107, QR-259). A secret is a [`Secret`]: it has no
//! `Display`, its `Debug` is redacted, and errors here carry the credential's *name*, never its content.

pub mod conformance;
pub use conformance::{assert_conforms, CredentialFixture, SetFn, Setup};

use std::collections::HashMap;
use std::fmt;
use std::path::{Component, Path, PathBuf};

/// A secret value. `Debug` is redacted; the only way to read it is [`Secret::expose`], at the point of use.
#[derive(Clone)]
pub struct Secret(String);

impl Secret {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    /// The secret itself. Call this only where it goes into a request header, and never log the result.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Secret(<redacted>)")
    }
}

/// Why a credential could not be loaded. Never contains the secret.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CredentialError {
    /// No credentials directory, or no file of that name.
    Unconfigured(String),
    /// The file exists but is empty (or only whitespace).
    Empty(String),
    /// The file is readable by group or others (mode must be 0600 or stricter).
    BadPermissions(String),
    /// The name is not a plain file name (path separators, `..`).
    BadName(String),
    Unreadable(String),
}

impl fmt::Display for CredentialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CredentialError::Unconfigured(n) => write!(f, "credential '{n}' is not configured"),
            CredentialError::Empty(n) => write!(f, "credential '{n}' is empty"),
            CredentialError::BadPermissions(n) => {
                write!(f, "credential '{n}' is accessible to group or others")
            }
            CredentialError::BadName(n) => write!(f, "'{n}' is not a valid credential name"),
            CredentialError::Unreadable(n) => write!(f, "credential '{n}' could not be read"),
        }
    }
}
impl std::error::Error for CredentialError {}

/// Where adapters get their secrets.
pub trait CredentialSource: Send + Sync {
    /// Read the credential called `name`. Called again on each use, so a rotated credential is picked up without a restart.
    fn get(&self, name: &str) -> Result<Secret, CredentialError>;
}

/// `$CREDENTIALS_DIRECTORY/<name>`, fail closed. The production source.
#[derive(Debug, Clone)]
pub struct DirCredentialSource {
    dir: PathBuf,
}

impl DirCredentialSource {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// From `$CREDENTIALS_DIRECTORY`. If it is not set there is no source at all (never a fallback to another place).
    pub fn from_environment() -> Result<Self, CredentialError> {
        std::env::var_os("CREDENTIALS_DIRECTORY")
            .filter(|d| !d.is_empty())
            .map(|d| Self::new(PathBuf::from(d)))
            .ok_or_else(|| CredentialError::Unconfigured("$CREDENTIALS_DIRECTORY".into()))
    }
}

fn plain_name(name: &str) -> bool {
    !name.is_empty()
        && Path::new(name).components().count() == 1
        && matches!(
            Path::new(name).components().next(),
            Some(Component::Normal(_))
        )
}

impl CredentialSource for DirCredentialSource {
    fn get(&self, name: &str) -> Result<Secret, CredentialError> {
        if !plain_name(name) {
            return Err(CredentialError::BadName(name.to_string()));
        }
        let path = self.dir.join(name);
        let meta = std::fs::metadata(&path).map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => CredentialError::Unconfigured(name.to_string()),
            _ => CredentialError::Unreadable(name.to_string()),
        })?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if meta.permissions().mode() & 0o077 != 0 {
                return Err(CredentialError::BadPermissions(name.to_string()));
            }
        }
        #[cfg(not(unix))]
        let _ = meta; // Windows uses its own credential mechanism (per-OS supervisor); not decided here
        let text = std::fs::read_to_string(&path)
            .map_err(|_| CredentialError::Unreadable(name.to_string()))?;
        let value = text.trim();
        if value.is_empty() {
            return Err(CredentialError::Empty(name.to_string()));
        }
        Ok(Secret::new(value))
    }
}

/// An in-memory source. **For tests and development only** (the `DEV_` exception): production never uses it.
#[derive(Debug, Default, Clone)]
pub struct InMemoryCredentials(pub HashMap<String, String>);

impl InMemoryCredentials {
    pub fn with(name: &str, value: &str) -> Self {
        Self(HashMap::from([(name.to_string(), value.to_string())]))
    }
}

impl CredentialSource for InMemoryCredentials {
    fn get(&self, name: &str) -> Result<Secret, CredentialError> {
        if !plain_name(name) {
            return Err(CredentialError::BadName(name.to_string()));
        }
        self.0
            .get(name)
            .filter(|v| !v.trim().is_empty())
            .map(|v| Secret::new(v.trim()))
            .ok_or_else(|| CredentialError::Unconfigured(name.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    fn write(dir: &Path, name: &str, body: &str, mode: u32) {
        use std::os::unix::fs::PermissionsExt;
        std::fs::write(dir.join(name), body).unwrap();
        std::fs::set_permissions(dir.join(name), std::fs::Permissions::from_mode(mode)).unwrap();
    }

    fn err_of(r: Result<Secret, CredentialError>) -> Option<CredentialError> {
        r.err()
    }

    #[test]
    fn a_secret_never_shows_in_debug_output_or_in_an_error() {
        let s = Secret::new("sk-ant-REAL-LOOKING-VALUE");
        assert!(!format!("{s:?}").contains("REAL"));
        let e = CredentialError::Empty("anthropic-api-key".into());
        assert!(!e.to_string().contains("sk-"));
    }

    #[cfg(unix)]
    #[test]
    fn the_directory_source_fails_closed() {
        let d = tempfile::tempdir().unwrap();
        let src = DirCredentialSource::new(d.path());
        assert_eq!(
            err_of(src.get("missing")),
            Some(CredentialError::Unconfigured("missing".into()))
        );
        write(d.path(), "empty", "  \n", 0o600);
        assert_eq!(
            err_of(src.get("empty")),
            Some(CredentialError::Empty("empty".into()))
        );
        write(d.path(), "loose", "value", 0o640);
        assert_eq!(
            err_of(src.get("loose")),
            Some(CredentialError::BadPermissions("loose".into()))
        );
        write(d.path(), "world", "value", 0o604);
        assert_eq!(
            err_of(src.get("world")),
            Some(CredentialError::BadPermissions("world".into()))
        );
        write(d.path(), "good", "the-value\n", 0o600);
        assert_eq!(
            src.get("good").unwrap().expose(),
            "the-value",
            "trailing newline trimmed"
        );
        write(d.path(), "ro", "v", 0o400);
        assert!(src.get("ro").is_ok(), "read-only owner-only is fine");
    }

    #[test]
    fn a_name_that_is_not_a_plain_file_name_is_refused() {
        let d = tempfile::tempdir().unwrap();
        let src = DirCredentialSource::new(d.path());
        for bad in ["", "../etc/passwd", "a/b", "/abs", ".."] {
            assert!(
                matches!(err_of(src.get(bad)), Some(CredentialError::BadName(_))),
                "{bad:?}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_rotated_credential_is_picked_up_without_a_restart() {
        let d = tempfile::tempdir().unwrap();
        let src = DirCredentialSource::new(d.path());
        write(d.path(), "k", "old", 0o600);
        assert_eq!(src.get("k").unwrap().expose(), "old");
        write(d.path(), "k", "new", 0o600);
        assert_eq!(src.get("k").unwrap().expose(), "new");
    }

    #[test]
    fn an_environment_variable_with_the_same_name_is_never_used() {
        // the in-memory source and the directory source have no env path at all; set one and prove it is ignored
        std::env::set_var("ANTHROPIC_TEST_CRED_NAME", "from-env");
        let d = tempfile::tempdir().unwrap();
        assert!(DirCredentialSource::new(d.path())
            .get("ANTHROPIC_TEST_CRED_NAME")
            .is_err());
        assert!(InMemoryCredentials::default()
            .get("ANTHROPIC_TEST_CRED_NAME")
            .is_err());
    }

    // ---- the shared contract, run against the kit's own sources ----

    type Shared = std::sync::Arc<std::sync::Mutex<HashMap<String, String>>>;

    struct InMemoryFixture;
    impl CredentialFixture for InMemoryFixture {
        fn name(&self) -> &'static str {
            "in-memory"
        }
        fn setup(&self, entries: &[(&str, &str)]) -> Setup {
            use std::sync::{Arc, Mutex};
            let store: Shared = Arc::new(Mutex::new(
                entries
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect(),
            ));
            struct Live(Shared);
            impl CredentialSource for Live {
                fn get(&self, name: &str) -> Result<Secret, CredentialError> {
                    InMemoryCredentials(self.0.lock().unwrap().clone()).get(name)
                }
            }
            let w = store.clone();
            Setup {
                source: Box::new(Live(store)),
                set: Box::new(move |k, v| {
                    w.lock().unwrap().insert(k.to_string(), v.to_string());
                }),
                guard: Box::new(()),
            }
        }
    }

    #[test]
    fn the_in_memory_source_keeps_the_contract() {
        assert_conforms(&InMemoryFixture);
    }

    #[cfg(unix)]
    struct DirFixture;
    #[cfg(unix)]
    impl CredentialFixture for DirFixture {
        fn name(&self) -> &'static str {
            "directory"
        }
        fn setup(&self, entries: &[(&str, &str)]) -> Setup {
            let d = tempfile::tempdir().unwrap();
            for (k, v) in entries {
                write(d.path(), k, v, 0o600);
            }
            let dir = d.path().to_path_buf();
            Setup {
                source: Box::new(DirCredentialSource::new(&dir)),
                set: Box::new(move |k, v| write(&dir, k, v, 0o600)),
                guard: Box::new(d),
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn the_directory_source_keeps_the_contract() {
        assert_conforms(&DirFixture);
    }

    // ---- the contract catches sources that break it (each of these is a real mistake someone could make) ----

    /// A source that wraps a good one and breaks exactly one property.
    #[derive(Clone, Copy, PartialEq)]
    enum Break {
        EnvFallback,
        LeaksInError,
        AcceptsPathNames,
        NoTrim,
        ReturnsBlank,
        CachesForever,
    }
    struct Broken(Break);
    impl CredentialFixture for Broken {
        fn name(&self) -> &'static str {
            "broken"
        }
        fn setup(&self, entries: &[(&str, &str)]) -> Setup {
            use std::sync::{Arc, Mutex};
            let kind = self.0;
            let store: Shared = Arc::new(Mutex::new(
                entries
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect(),
            ));
            let first_seen: Shared = Arc::default();
            struct Src(Break, Shared, Shared);
            impl CredentialSource for Src {
                fn get(&self, name: &str) -> Result<Secret, CredentialError> {
                    let kind = self.0;
                    if kind == Break::EnvFallback {
                        if let Ok(v) = std::env::var(name.replace('-', "_").to_uppercase()) {
                            return Ok(Secret::new(v));
                        }
                    }
                    if !plain_name(name) && kind != Break::AcceptsPathNames {
                        // a leaky error names the credential's value in the bad-name message
                        let leak = self
                            .1
                            .lock()
                            .unwrap()
                            .values()
                            .max_by_key(|v| v.len())
                            .cloned()
                            .unwrap_or_default();
                        return Err(CredentialError::BadName(if kind == Break::LeaksInError {
                            format!("{name} (store holds: {leak})")
                        } else {
                            name.to_string()
                        }));
                    }
                    let store = self.1.lock().unwrap();
                    let Some(v) = store.get(name).cloned() else {
                        return Err(CredentialError::Unconfigured(name.to_string()));
                    };
                    if kind != Break::ReturnsBlank && v.trim().is_empty() {
                        return Err(CredentialError::Empty(name.to_string()));
                    }
                    let v = if kind == Break::NoTrim {
                        v
                    } else {
                        v.trim().to_string()
                    };
                    if kind == Break::CachesForever {
                        let mut seen = self.2.lock().unwrap();
                        return Ok(Secret::new(
                            seen.entry(name.to_string()).or_insert(v).clone(),
                        ));
                    }
                    Ok(Secret::new(v))
                }
            }
            let w = store.clone();
            Setup {
                source: Box::new(Src(kind, store, first_seen)),
                set: Box::new(move |k, v| {
                    w.lock().unwrap().insert(k.to_string(), v.to_string());
                }),
                guard: Box::new(()),
            }
        }
    }

    #[test]
    #[should_panic(expected = "environment variable must never supply")]
    fn the_contract_catches_an_environment_fallback() {
        assert_conforms(&Broken(Break::EnvFallback));
    }
    #[test]
    #[should_panic(expected = "an error's Display contains the stored value")]
    fn the_contract_catches_a_value_leaked_into_an_error() {
        assert_conforms(&Broken(Break::LeaksInError));
    }
    #[test]
    #[should_panic(expected = "must be BadName")]
    fn the_contract_catches_a_source_that_accepts_path_names() {
        assert_conforms(&Broken(Break::AcceptsPathNames));
    }
    #[test]
    #[should_panic(expected = "surrounding whitespace must be trimmed")]
    fn the_contract_catches_untrimmed_values() {
        assert_conforms(&Broken(Break::NoTrim));
    }
    #[test]
    #[should_panic(expected = "never a usable secret")]
    fn the_contract_catches_a_blank_credential_returned_as_a_secret() {
        assert_conforms(&Broken(Break::ReturnsBlank));
    }
    #[test]
    #[should_panic(expected = "rotated credential must be seen")]
    fn the_contract_catches_a_source_that_caches_forever() {
        assert_conforms(&Broken(Break::CachesForever));
    }

    #[test]
    fn no_credentials_directory_means_no_source() {
        // run in a scrubbed copy of the check rather than mutating the process environment for other tests
        let r = std::thread::spawn(|| {
            let saved = std::env::var_os("CREDENTIALS_DIRECTORY");
            std::env::remove_var("CREDENTIALS_DIRECTORY");
            let r = DirCredentialSource::from_environment().is_err();
            if let Some(v) = saved {
                std::env::set_var("CREDENTIALS_DIRECTORY", v);
            }
            r
        })
        .join()
        .unwrap();
        assert!(r);
    }
}
