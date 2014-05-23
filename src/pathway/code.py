

import networkx


#
# read the graph
#
def read_graph():
    g = networkx.read_edgelist("ucscSuperV3.0p2_pathway.tab", create_using=networkx.DiGraph(), delimiter="\t", data=[("edge", str)])
    return g


#
#
#
def gen_filename(n_op, e_type):
    fn = "{0}_{1:03d}.tab".format(e_type, n_op)
    return fn


#
#
#
def gen_code(d_fh, e_type):
    #
    name = e_type.upper()
    rel = {"complex":"&", "family":"|"} [e_type]

    #
    # first, output the model
    #
    for n_op in d_fh:
        # add predicate for # of operands
        pred = "{0}_{1:03d}".format(name, n_op)
        print "// BEGIN: {0}".format(pred)
        str_op = ", ".join("ArgumentType.UniqueID" for i in range(1, n_op+2)) # # of arguments = n_op + 1
        print "model.add predicate: \"{0}\", types:[{1}]".format(pred, str_op)
        #  rule for # of operands
        str_op_1 = ", ".join("G{0:03d}".format(i) for i in range(1, n_op+1)) # "G1, G2, G3 [...]"
        str_op_2 = " {0} ".format(rel).join("Active(G{0:03d}, P)".format(i) for i in range(1, n_op+1)) # "Active(G1, P) & Active(G2, P) [...]"
        print "model.add rule : ({0}({1}, G_OUT) & ({2})) >> Active(G_OUT, P), weight : 1".format(pred, str_op_1, str_op_2)
        print "// END: {0}".format(pred)
        print ""

    #
    # then, load groundings from files
    #
    for n_op in d_fh:
        pred = "{0}_{1:03d}".format(name, n_op)
        fn = gen_filename(n_op, e_type)
        print "// BEGIN: {0}".format(pred)
        print "inserter = data.getInserter({0}, dummy_tr)".format(pred)
        print "InserterUtils.loadDelimitedData(inserter, traindir + \"{0}\", \"\\t\")".format(fn)
        print "// END: {0}".format(pred)
        print ""


#
# generate PSL code & data for composite entities: complexes / families
# act like AND/OR gates w/ various number of operands
#
def gen_gates(g, e_type="complex"):
    # file handles (each file stores expressions w/ a specific number of operands)
    d_fh = {}
    # generate rules for entities
    edgeTypes = networkx.get_edge_attributes(g, "edge")
    nc = 0
    for n in g.nodes():
        if n.endswith("({0})".format(e_type)):
            #
            if len(g.successors(n)) > 0 and len(g.predecessors(n)) > 0:
                # select only component> predecessors
                l_pc = [p for p in g.predecessors(n) if edgeTypes[(p, n)] == "component>"]
                # add
                nc = nc + 1
                # output
                n_op = len(l_pc)
                if n_op <= 0:
                    continue
                # open file for # of operands in this rule
                if n_op not in d_fh:
                    fn = gen_filename(n_op, e_type)
                    fh = open(fn, "w")
                    d_fh[n_op] = fh
                else:
                    fh = d_fh[n_op]
                #
                fh.write("{0}\t{1}\n".format("\t".join(l_pc), n))
    #
    print "number of entities: {0}".format(nc)
    # close all
    for fh in d_fh.values():
        fh.close()
    # output Groovy code for PSL
    gen_code(d_fh, e_type)
    #


#
#
#
if __name__ == '__main__':
    g = read_graph()
    gen_gates(g)
