# strip_wikitext.awk — the plain text of MediaWiki pages (used by get_text.sh).
# Run with LC_ALL=C (byte-wise, so any awk handles the UTF-8 the same way).
# Drops templates, files, categories, link targets, bold/italic quotes and
# tags; prints one paragraph per line.
{ buf = buf $0 "\n" }
END {
    while (gsub(/\{\{[^{}]*\}\}/, "", buf)) {}
    gsub(/\[\[(ملف|صورة|تصنيف|File|Image|Category):[^]\n]*\]\]/, "", buf)
    gsub(/\[\[[^]|\n]*\|/, "", buf)
    gsub(/\[\[|\]\]/, "", buf)
    gsub(/'''|''/, "", buf)
    gsub(/<[^>\n]*>/, "", buf)
    gsub(/&nbsp;/, " ", buf)
    n = split(buf, line, "\n")
    for (i = 1; i <= n; i++) {
        s = line[i]
        gsub(/\t|\r/, " ", s)
        sub(/^[=*#:;| ]+/, "", s)
        sub(/[= ]+$/, "", s)
        gsub(/  +/, " ", s)
        if (s != "") print s
    }
}
