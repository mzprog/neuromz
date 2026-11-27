/* test_text.c — UTF-8, vocabularies, text datasets (incl. Arabic). */
#include "test.h"

#include <stdlib.h>
#include <string.h>

static void write_bytes(const char *path, const char *bytes, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fwrite(bytes, 1, n, f);
    fclose(f);
}

static void test_utf8(void) {
    /* "محمد" = U+0645 U+062D U+0645 U+062F, 2 bytes each */
    const char *ar = "\xd9\x85\xd8\xad\xd9\x85\xd8\xaf";
    const char *mixed = "a\xc3\xa9\xe2\x82\xac\xf0\x9f\x98\x80"; /* a é € 😀 */
    uint32_t *cp = NULL;
    size_t n = 0, bad = 0, i;
    char buf[5];
    CHECK_OK(nmz_utf8_decode(ar, strlen(ar), &cp, &n, &bad));
    CHECK(n == 4 && bad == 0 && cp[0] == 0x645 && cp[1] == 0x62D && cp[3] == 0x62F);
    free(cp);
    CHECK_OK(nmz_utf8_decode(mixed, strlen(mixed), &cp, &n, &bad));
    CHECK(n == 4 && bad == 0 && cp[0] == 'a' && cp[1] == 0xE9 && cp[2] == 0x20AC && cp[3] == 0x1F600);
    for (i = 0; i < n; i++) CHECK(nmz_utf8_encode(cp[i], buf) == (int)(i + 1)); /* 1,2,3,4 bytes */
    free(cp);
    {
        /* invalid: stray continuation, overlong '/', truncated €, surrogate, > U+10FFFF, 0xFF */
        static const char bad_bytes[] = "x\x80y\xc0\xafz\xe2\x82\xed\xa0\x80\xf4\x90\x80\x80\xff!";
        CHECK_OK(nmz_utf8_decode(bad_bytes, sizeof bad_bytes - 1, &cp, &n, &bad));
        CHECK(cp[0] == 'x' && cp[1] == NMZ_REPLACEMENT_CHAR && cp[2] == 'y' && cp[n - 1] == '!');
        CHECK(bad == 1 + 2 + 2 + 3 + 4 + 1); /* one U+FFFD per invalid byte */
        for (i = 0; i < n; i++) CHECK(cp[i] == NMZ_REPLACEMENT_CHAR || (cp[i] < 0x80));
        free(cp);
    }
    CHECK(nmz_utf8_encode(0xD800, buf) == 0 && nmz_utf8_encode(0x110000, buf) == 0);
    CHECK(nmz_utf8_encode(0x645, buf) == 2 && !memcmp(buf, "\xd9\x85", 3));
}

static void test_vocab(void) {
    uint32_t text[8] = {'b', 0x645, 'a', 'b', '\n', 0x645, 0x645, 'a'};
    nmz_vocab *v = NULL, *c = NULL;
    CHECK_OK(nmz_vocab_build(&v, text, 8));
    if (!v) return;
    CHECK(v->size == 4);
    CHECK(v->codepoints[0] == '\n' && v->codepoints[1] == 'a' && v->codepoints[3] == 0x645); /* sorted */
    CHECK(v->counts[3] == 3 && v->counts[1] == 2 && v->counts[0] == 1);
    CHECK(nmz_vocab_index(v, 'b') == 2 && nmz_vocab_index(v, 'z') == -1);
    CHECK_OK(nmz_vocab_copy(&c, v));
    CHECK(c && c->size == 4 && c->counts[3] == 3);
    nmz_vocab_free(c);
    c = NULL;
    CHECK_STATUS(nmz_vocab_create(&c, 2, text + 1, NULL, 0), NMZ_ERR_ARG); /* U+0645 then 'a': not sorted */
    CHECK(c == NULL);
    nmz_vocab_free(v);
    CHECK_STATUS(nmz_vocab_build(&v, text, 0), NMZ_ERR_FORMAT);
}

static void test_text_dataset_arabic(void) {
    /* items mode: CR LF, an empty line, trailing spaces, a BOM */
    const char *p = test_path("t_names_ar.txt");
    const char *txt = "\xef\xbb\xbf\xd9\x85\xd8\xad\xd9\x85\xd8\xaf\r\n\r\n\xd8\xb9\xd9\x84\xd9\x8a  \n"
                      "\xd8\xb9\xd9\x85\xd8\xb1";                      /* محمد, علي, عمر */
    nmz_text_opts o = nmz_text_defaults();
    nmz_dataset *ds = NULL;
    int i, nl;
    write_bytes(p, txt, strlen(txt));
    o.items = 1;
    o.seq_len = 3;
    CHECK_OK(nmz_dataset_load_text(&ds, p, &o));
    if (!ds) return;
    /* "\nمحمد\nعلي\nعمر\n" = 1+4+1+3+1+3+1 = 14 code points */
    CHECK_MSG(ds->n_tokens == 14, "%zu tokens", ds->n_tokens);
    CHECK(ds->invalid_utf8 == 0 && ds->vocab->items == 1);
    nl = nmz_vocab_index(ds->vocab, '\n');
    CHECK(nl == 0 && ds->vocab->counts[nl] == 4);
    CHECK(ds->vocab->size == 1 + 7); /* \n م ح د ع ل ي ر */
    CHECK(ds->time_steps == 3 && ds->y_index == 1 && ds->n_inputs == 3 && ds->n_outputs == ds->vocab->size);
    CHECK(ds->n_samples == (14 - 1) / 3);
    CHECK(ds->x[0] == (float)nl); /* the text starts with the end-of-item token */
    for (i = 0; i + 1 < ds->n_samples * 3; i++) CHECK(ds->y[i] == ds->x[i + 1]); /* target = next character */
    CHECK(ds->x[1] == (float)nmz_vocab_index(ds->vocab, 0x645));
    {
        /* encoding with a given vocabulary; an unknown character is a clear error */
        nmz_dataset *d2 = NULL;
        const char *q = test_path("t_names_ar2.txt");
        write_bytes(q, "\xd8\xb9\xd9\x84\xd9\x8a\nzz\n", 9);
        o.vocab = ds->vocab;
        CHECK_STATUS(nmz_dataset_load_text(&d2, q, &o), NMZ_ERR_FORMAT);
        CHECK_MSG(strstr(nmz_last_error(), "U+007A") != NULL, "%s", nmz_last_error());
        write_bytes(q, "\xd8\xb9\xd9\x84\xd9\x8a\n\xd8\xb9\xd9\x85\xd8\xb1\n", 13);
        CHECK_OK(nmz_dataset_load_text(&d2, q, &o));
        CHECK(d2 && d2->vocab->size == ds->vocab->size && d2->vocab->counts[nl] == 3);
        nmz_dataset_free(d2);
    }
    nmz_dataset_free(ds);
}

static void test_text_dataset_errors(void) {
    const char *p = test_path("t_text_bad.txt");
    nmz_text_opts o = nmz_text_defaults();
    nmz_dataset *ds = NULL;
    write_bytes(p, "ab\xff\xfe" "cdefgh", 10);
    o.seq_len = 4;
    CHECK_OK(nmz_dataset_load_text(&ds, p, &o));
    CHECK(ds && ds->invalid_utf8 == 2 && nmz_vocab_index(ds->vocab, NMZ_REPLACEMENT_CHAR) >= 0);
    nmz_dataset_free(ds);
    ds = NULL;
    o.seq_len = 50;
    CHECK_STATUS(nmz_dataset_load_text(&ds, p, &o), NMZ_ERR_FORMAT);
    CHECK(strstr(nmz_last_error(), "not enough") != NULL);
    CHECK_STATUS(nmz_dataset_load_text(&ds, test_path("missing.txt"), &o), NMZ_ERR_IO);
    o.seq_len = 0;
    CHECK_STATUS(nmz_dataset_load_text(&ds, p, &o), NMZ_ERR_ARG);
    CHECK(ds == NULL);
}


/* optional Arabic normalisation: tashkeel + tatweel, alef forms */
static void test_arabic_normalize(void) {
    /* "مُحَمَّدٌ كـــتاب" (with harakat, shadda, tanwin, tatweel) "أحمد إبراهيم آمنة ٱلله" */
    const char *in = "\xd9\x85\xd9\x8f\xd8\xad\xd9\x8e\xd9\x85\xd9\x91\xd9\x8e\xd8\xaf\xd9\x8c \xd9\x83\xd9\x80\xd9\x80"
                     "\xd9\x80\xd8\xaa\xd8\xa7\xd8\xa8 \xd8\xa3\xd8\xad\xd9\x85\xd8\xaf \xd8\xa5\xd8\xa8\xd8\xb1\xd8\xa7\xd9\x87"
                     "\xd9\x8a\xd9\x85 \xd8\xa2\xd9\x85\xd9\x86\xd8\xa9 \xd9\xb1\xd9\x84\xd9\x84\xd9\x87";
    static const uint32_t strip[] = {0x645, 0x62D, 0x645, 0x62F, ' ', 0x643, 0x62A, 0x627, 0x628, ' ', 0x623, 0x62D,
                                     0x645, 0x62F, ' ', 0x625, 0x628, 0x631, 0x627, 0x647, 0x64A, 0x645, ' ', 0x622,
                                     0x645, 0x646, 0x629, ' ', 0x671, 0x644, 0x644, 0x647};
    uint32_t *cp = NULL;
    size_t n = 0, bad = 0, m, i;
    CHECK_OK(nmz_utf8_decode(in, strlen(in), &cp, &n, &bad));
    if (!cp) return;
    CHECK(nmz_text_normalize(cp, n, 0) == n); /* off: unchanged */
    m = nmz_text_normalize(cp, n, NMZ_TEXT_STRIP_DIACRITICS);
    CHECK_MSG(m == sizeof strip / sizeof strip[0] && n - m == 8, "%zu -> %zu", n, m); /* 5 marks + 3 tatweel */
    for (i = 0; i < m && i < sizeof strip / sizeof strip[0]; i++) CHECK_MSG(cp[i] == strip[i], "[%zu] U+%04X", i, cp[i]);
    m = nmz_text_normalize(cp, m, NMZ_TEXT_UNIFY_ALEF); /* أ إ آ ٱ → ا, nothing removed */
    CHECK(m == sizeof strip / sizeof strip[0] && cp[10] == 0x627 && cp[15] == 0x627 && cp[23] == 0x627 && cp[28] == 0x627);
    CHECK(cp[2] == 0x645 && cp[18] == 0x627); /* other letters untouched */
    free(cp);
    {
        /* a dataset with the flags: vocabulary without the marks, flags kept, removals counted */
        const char *path = test_path("t_tashkeel.txt");
        nmz_text_opts o = nmz_text_defaults();
        nmz_dataset *plain = NULL, *norm = NULL;
        write_bytes(path, in, strlen(in));
        o.seq_len = 4;
        CHECK_OK(nmz_dataset_load_text(&plain, path, &o));
        o.normalize = NMZ_TEXT_NORMALIZE_ALL;
        CHECK_OK(nmz_dataset_load_text(&norm, path, &o));
        if (plain && norm) {
            CHECK(plain->vocab->normalize == 0 && plain->n_removed == 0);
            CHECK(norm->vocab->normalize == NMZ_TEXT_NORMALIZE_ALL && norm->n_removed == 8);
            CHECK(nmz_vocab_index(plain->vocab, 0x64F) >= 0 && nmz_vocab_index(norm->vocab, 0x64F) < 0); /* damma */
            CHECK(nmz_vocab_index(norm->vocab, 0x640) < 0 && nmz_vocab_index(norm->vocab, 0x623) < 0);
            CHECK_MSG(plain->vocab->size - norm->vocab->size == 9, /* ُ َ ّ ٌ ـ and أ إ آ ٱ */ "%d vs %d", plain->vocab->size, norm->vocab->size);
            /* encoding with the model's vocabulary applies the model's flags */
            o.normalize = 0;
            o.vocab = norm->vocab;
            nmz_dataset_free(plain);
            plain = NULL;
            CHECK_OK(nmz_dataset_load_text(&plain, path, &o));
            CHECK(plain && plain->n_removed == 8 && plain->n_samples == norm->n_samples);
        }
        nmz_dataset_free(plain);
        nmz_dataset_free(norm);
    }
}

const test_case text_tests[] = {
    {"text/utf8_arabic_and_invalid", test_utf8},
    {"text/vocab_codepoints", test_vocab},
    {"text/dataset_arabic_items", test_text_dataset_arabic},
    {"text/dataset_errors", test_text_dataset_errors},
    {"text/arabic_normalization", test_arabic_normalize},
    {NULL, NULL}};
