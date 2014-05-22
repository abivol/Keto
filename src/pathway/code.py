
#
# read the graph
#
def read_graph():
    g = networkx.read_edgelist("ucscSuperV3.0p2_pathway.tab", create_using=networkx.DiGraph(), delimiter="\t", data=[("edge", str)])
    return g


#
# generate PSL code & data for complexes
# (act like AND gates w/ various number of operands)
#
def gen_complexes(g):
    # file handles (each file stores expressions w/ a specific number of operands)
    d_fh = {}
    # generate rules for complexes
    edgeTypes = networkx.get_edge_attributes(g, "edge")
    nc = 0
    for n in g.nodes():
        if n.endswith("(complex)"):
            #
            if len(g.successors(n)) > 0 and len(g.predecessors(n)) > 0:
                # select only component> predecessors
                l_pc = [p for p in g.predecessors(n) if edgeTypes[(p, n)] == "component>"]
                # add
                nc = nc + 1
                # output
                np = len(l_pc)
                if np not in d_fh:
                    fn = "complexes_{%"
                print n + " : " + ",".join(l_pc)
    #
    print "number of complexes: {0}".format(nc)
