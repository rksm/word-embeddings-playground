use anyhow::Result;
use std::path::Path;

pub fn backup_file(file: impl AsRef<Path>, max_backups: u8) -> Result<()> {
    // backups
    let file = file.as_ref();
    if let Some(dir) = file.parent() {
        let fname = file.file_name().unwrap().to_string_lossy();
        let mut prev_files = std::fs::read_dir(dir)?
            .filter_map(|f| f.ok())
            .filter_map(|f| {
                let n = f.file_name();
                let n = n.to_string_lossy();
                if !n.starts_with(&*fname) {
                    return None;
                }
                let n = n.trim_start_matches(&*fname);
                if n.is_empty() {
                    return None;
                }
                let n = n.trim_start_matches('.');
                if n.is_empty() {
                    return None;
                }
                let n = n.parse::<u8>().ok()?;
                Some((n, f.path()))
            })
            .collect::<Vec<_>>();
        prev_files.sort_by_key(|(n, _)| *n);
        prev_files.reverse();
        for (n, f) in prev_files {
            if n > max_backups {
                std::fs::remove_file(f)?;
                continue;
            }
            let new_name = format!("{}.{}", fname, n + 1);
            std::fs::rename(f, dir.join(new_name))?;
        }
    }

    if file.exists() {
        std::fs::rename(file, file.with_extension("nn.1"))?;
    }

    Ok(())
}
