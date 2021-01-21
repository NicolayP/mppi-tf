#include "utile.hpp"

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/summary/summary_file_writer.h"

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

Output utile::blockDiag(Scope root, Input in, int nb) {
    auto pad = ZerosLike(root, in);
    Output hori_el, tmp;

    initializer_list<Input> list_init;
    //auto first = Identity(root, in);
    for (int i=0; i<nb; i++) {
        for (int j=0; j<nb; j++) {
            if (i == 0 && j == 0) {
                hori_el = Identity(root, in);
            } else if (j == 0) {
                hori_el = Identity(root, pad);
            } else if (i == j) {
                list_init = {hori_el, in};
                InputList newlist(list_init);
                hori_el = Concat(root, newlist, 1);
            } else {
                list_init = {hori_el, pad};
                InputList newlist(list_init);
                hori_el = Concat(root, newlist, 1);
            }
        }

        if (i == 0) {
            tmp = Identity(root, hori_el);
        } else {
            list_init = {tmp, hori_el};
            InputList newlist(list_init);
            tmp = Concat(root, newlist, 0);
        }
    }

    return tmp;
}


Status utile::logGraph(Scope root) {
    GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
    SummaryWriterInterface* w;
    TF_CHECK_OK(CreateSummaryFileWriter(1, 0, "../graphs",
                                        ".img-graph", Env::Default(), &w));
    TF_CHECK_OK(w->WriteGraph(0, make_unique<GraphDef>(graph)));
    return Status::OK();
}
