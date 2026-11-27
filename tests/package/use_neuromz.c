/* use_neuromz.c — trains XOR through an installed NeuroMZ (find_package). */
#include <stdio.h>

#include <neuromz.h>

int main(void) {
    static const float X[8] = {0, 0, 0, 1, 1, 0, 1, 1}, Y[4] = {0, 1, 1, 0};
    int sizes[3] = {2, 4, 1};
    nmz_dataset *ds = NULL;
    nmz_net *net = NULL;
    nmz_train_result r;
    if (nmz_dataset_create(&ds, 4, 2, 1, X, Y) != NMZ_OK || nmz_net_create_quick(&net, 3, sizes, 42, NULL) != NMZ_OK ||
        nmz_net_train(net, ds, NULL, &r) != NMZ_OK) {
        fprintf(stderr, "error: %s\n", nmz_last_error());
        return 1;
    }
    printf("NeuroMZ %s: XOR %s after %d epochs\n", NMZ_VERSION_STRING, r.converged ? "learned" : "NOT learned", r.epochs);
    nmz_net_free(net);
    nmz_dataset_free(ds);
    return r.converged ? 0 : 1;
}
