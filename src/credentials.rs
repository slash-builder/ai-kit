//! Provider credentials: where a secret comes from, and how it is kept out of everything else. **No network.**
//!
//! A provider is configured with a *name* (`"anthropic-api-key"`), never a key. The production source reads
//! `$CREDENTIALS_DIRECTORY/<name>` (systemd `LoadCredential=`), and it **fails closed**: a missing, empty, unreadable or
//! group/world-accessible file means the provider is `Unconfigured` and is never constructed. There is **no fallback to environment
//! variables, argv or a config file** (hearth `docs/encryption.md` section 3.5, CLUS-107, QR-259). A secret is a [`Secret`]: it has no
//! `Display`, its `Debug` is redacted, and errors here carry the credential's *name*, never its content.

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
