#!/bin/bash
# Verify that the full-sources JAR contains all expected source files.
# Usage: ./scripts/verify-full-sources-jar.sh [path-to-jar]
#
# Works whether MMCoreJ_wrap is its own repo (for future) or a subdirectory of
# mmCoreAndDevices.

set -e

JAR_FILE="${1:-target/MMCoreJ-*-full-sources.jar}"

# Files that may be omitted from the JAR despite being in git.
ALLOWED_MISSING=(
    'Makefile.am'
    'build.xml'
    '*.vcxproj'
    '*.vcxproj.filters'
)

# Subprojects to check for source files.
SUBPROJECTS=(
    'subprojects/mmcore'
    'subprojects/mmdevice'
)

# Expand glob if needed
JAR_FILE=$(echo $JAR_FILE)

if [[ ! -f "$JAR_FILE" ]]; then
    echo "ERROR: JAR file not found: $JAR_FILE"
    echo "Run 'mvn package' first."
    exit 1
fi

echo "Verifying: $JAR_FILE"

# Verify subprojects exist on filesystem (required as reference for JAR verification)
for subproj in "${SUBPROJECTS[@]}"; do
    if [[ ! -d "$subproj" ]]; then
        echo "ERROR: Required subproject directory not found: $subproj"
        echo "Run 'meson setup' before 'mvn package'."
        exit 1
    fi
    if [[ ! -f "$subproj/meson.build" ]]; then
        echo "ERROR: Subproject missing meson.build: $subproj"
        echo "Subproject may be incomplete or corrupted."
        exit 1
    fi
done

# Create temp files for comparison
JAR_CONTENTS=$(mktemp)
EXPECTED_FILES=$(mktemp)
trap "rm -f $JAR_CONTENTS $EXPECTED_FILES" EXIT

# Detect if we're in a subdirectory of a larger git repo.
# If so, we need to strip the prefix from git ls-files output.
# Empty if at root, "subdir/" if in subdir:
GIT_PREFIX=$(git rev-parse --show-prefix)

# List JAR contents (excluding META-INF/)
jar tf "$JAR_FILE" | grep -v 'META-INF' | sort > "$JAR_CONTENTS"

# Helper to strip the git prefix from paths
strip_prefix() {
    if [[ -n "$GIT_PREFIX" ]]; then
        sed "s|^${GIT_PREFIX}||"
    else
        cat
    fi
}

# Helper to filter out allowed missing files
filter_allowed_missing() {
    local result
    result=$(cat)
    for pattern in "${ALLOWED_MISSING[@]}"; do
        # Convert glob pattern to grep -v pattern
        # e.g., *.vcxproj -> \.vcxproj$
        local regex=$(echo "$pattern" | sed 's/\./\\./g' | sed 's/\*/.*/g')
        result=$(echo "$result" | grep -v "$regex" || true)
    done
    echo "$result"
}

# Build expected file list from git
{
    # All files tracked by git in this directory
    git ls-files --full-name | strip_prefix

    # subprojects (from submodules - have their own .git)
    for subproj in "${SUBPROJECTS[@]}"; do
        if [[ -d "$subproj" ]]; then
            if [[ -e "$subproj/.git" ]]; then
                # Submodule: query its own git
                git -C "$subproj" ls-files --full-name | sed "s|^|$subproj/|"
            else
                # Not a git repo: enumerate all files on disk
                # (meson.build existence already verified above)
                find "$subproj" -type f
            fi
        fi
    done
} | sort -u > "$EXPECTED_FILES"

# Compare (filtering out allowed missing files)
MISSING=$(comm -23 "$EXPECTED_FILES" "$JAR_CONTENTS" | filter_allowed_missing)

if [[ -n "$MISSING" ]]; then
    echo "ERROR: The following expected source files are missing from the JAR:"
    echo "$MISSING"
    exit 1
fi

echo "OK: All expected source files are present in the JAR."
# Optionally show extra files in JAR (not an error, just informational)
# Filter out directory entries (end with /) and blank lines
EXTRA=$(comm -13 "$EXPECTED_FILES" "$JAR_CONTENTS" | grep -v '/$' | grep -v '^$' || true)
if [[ -n "$EXTRA" ]]; then
    echo "Note: JAR contains additional files not in git (this is OK):"
    echo "$EXTRA"
fi

exit 0
