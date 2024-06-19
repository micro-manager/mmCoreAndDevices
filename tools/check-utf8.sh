#!/bin/bash

invalid_found=0

# We would like to redirect iconv output to /dev/null, but the macOS version of
# iconv has problems with this ("iconv: iconv(): Inappropriate ioctl for
# device", sporadically for particular inputs). So use a regular file instead.
tmpfile=/tmp/check-utf8-temp
trap "rm -f $tmpfile" EXIT

for file in $(git ls-files | grep -E '\.(cpp|h|txt)$'); do
    if ! iconv -f UTF-8 "$file" >$tmpfile; then
        echo "Not valid UTF-8: $file" >&2
        invalid_found=1
    fi
done

[ $invalid_found = 0 ] && echo "All checked files are valid UTF-8" >&2

exit $invalid_found
