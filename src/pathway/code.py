#!/usr/bin/env python

#
# Convert superpathway format to PSL rules.
#


import networkx


#
#
#
def gen_filename(n_op, e_type):
    fn = "{0}_{1:03d}.tab".format(e_type, n_op)
    return fn


class PSLCodeGen:

    def __init__(self):
        self.PathwayFileName = None
        self.Graph = None
        self.d_NodeId = {}


    #
    # read the graph
    #
    def read_graph(self, fn):
        self.Graph = networkx.read_edgelist(fn,
            create_using=networkx.DiGraph(),
            delimiter="\t",
            data=[("edge", str)]
            )
        self.PathwayFileName = fn


    def get_node_id(self, n):
        #
        if n in self.d_NodeId:
            return self.d_NodeId[n]
        #
        lmax = 255
        if len(n) > lmax:
            j = "..."
            ls = int((lmax - len(j))/2)
            s = "{0}{1}{2}".format(n[0:ls], j, n[len(n)-ls:len(n)])
        else:
            s = n
        #
        self.d_NodeId[n] = s
        #
        return s


    #
    # generate code for composite entities: complex, family
    #
    def gen_code_gates(self, d_fh, e_type):
        #
        name = e_type.upper()
        rel = {"Complex":"&", "Family":"|"} [e_type]
        #
        fn = "PSL_{0}.groovy".format(e_type)
        fh = open(fn, "w")
        #
        # first, output the model
        #
        for n_op in d_fh:
            # add predicate for # of operands
            pred = "{0}_{1:03d}".format(name, n_op)
            fh.write("// BEGIN: {0}\n".format(pred))
            str_op = ", ".join("ArgumentType.UniqueID" for i in range(1, n_op+2)) # # of arguments = n_op + 1
            fh.write("model.add predicate: \"{0}\", types:[{1}]\n".format(pred, str_op))
            #  rule for # of operands
            str_op_1 = ", ".join("G{0:03d}".format(i) for i in range(1, n_op+1)) # "G1, G2, G3 [...]"
            str_op_2 = " {0} ".format(rel).join("Active(G{0:03d}, P)".format(i) for i in range(1, n_op+1)) # "Active(G1, P) & Active(G2, P) [...]"
            fh.write("model.add rule : ({0}({1}, G_OUT) & ({2})) >> Active(G_OUT, P), weight : 1\n".format(pred, str_op_1, str_op_2))
            fh.write("// END: {0}\n".format(pred))
            fh.write("\n")

        #
        # then, load groundings from files
        #
        for n_op in d_fh:
            pred = "{0}_{1:03d}".format(name, n_op)
            fn = gen_filename(n_op, e_type)
            fh.write("// BEGIN: {0}\n".format(pred))
            fh.write("inserter = data.getInserter({0}, dummy_tr)\n".format(pred))
            fh.write("InserterUtils.loadDelimitedData(inserter, traindir + \"{0}\", \"\\t\")\n".format(fn))
            fh.write("// END: {0}\n".format(pred))
            fh.write("\n")
        #
        fh.close()

    #
    # generate PSL code & data for composite entities: complexes / families,
    # which act like AND/OR gates w/ various number of operands
    #
    def parse_gates(self, e_type="Complex"):
        #
        g = self.Graph
        rel = {"Complex":"component>", "Family":"member>"} [e_type]
        # file handles (each file stores expressions w/ a specific number of operands)
        fn = "{0}.tab".format(e_type)
        fh = open(fn, "w")
        # generate rules for entities
        edgeTypes = networkx.get_edge_attributes(g, "edge")
        for n in g.nodes():
            if (not n.endswith("({0})".format(e_type.lower())) or
                len(g.successors(n)) <= 0 or len(g.predecessors(n)) <= 0):
                continue
            # select only component> predecessors
            l_pc = [p for p in g.predecessors(n) if edgeTypes[(p, n)] == rel]
            # output
            n_op = len(l_pc)
            if n_op <= 0:
                continue
            # weight
            w = 1.0
            if e_type == "Complex":
                w = 1.0 / n_op
            #
            for p in l_pc:
                fh.write("{0}\t{1}\t{2}\n".format(self.get_node_id(p), self.get_node_id(n), w))
        #
        fh.close()


    #
    # generate PSL code & data for 2 operand pathway relations
    #
    # e_type = "Activates" (G1, G2) or "Inhibits" (G1, G2)
    #
    def parse_2oper(self, e_type):
        #
        g = self.Graph
        rel = {"Activates":["-a>", "-t>", "-ap>"], "Inhibits":["-a|", "-t|", "-ap|"]} [e_type]
        #
        fn = "{0}.tab".format(e_type)
        fh = open(fn, "w")
        # generate rules for entities
        edgeTypes = networkx.get_edge_attributes(g, "edge")
        #
        for n in g.nodes():
            # select only matching successors
            l_sc = [s for s in g.successors(n) if edgeTypes[(n, s)] in rel]
            for s in l_sc:
                # output
                fh.write("{0}\t{1}\n".format(self.get_node_id(n), self.get_node_id(s)))
        #
        fh.close()



    #
    # select a smaller graph
    #
    def sel_graph(self, nodes, p_len=10, fn="graph_flt.tab"):
        g = self.Graph
        seln = set()
        #
        nodes = set(nodes).intersection(g.nodes())
        print "len(nodes): {0}".format(len(nodes))
        res = set()
        # enumerate all shortest paths of length up to p_len
        for i in nodes:
            print "node: {0}".format(i)
            ssp = networkx.single_source_shortest_path(g, i, p_len)
            target = set(ssp.keys()).intersection(nodes)
            # print "ssp: {0}".format(ssp)
            for t in target:
                p = ssp[t]
                print p
                # print "len(p): {0}".format(len(p))
                # add
                for j in p:
                    seln.add(j)
                print "ADD: len(seln): {0}".format(len(seln))
        #
        rg = g.subgraph(seln)
        #
        networkx.write_edgelist(rg, fn, delimiter="\t", data=["edge"], )
        return(rg)

#
#
#
def parse_graph():
    pcg = PSLCodeGen()
    pcg.read_graph("graph_flt.tab")
    pcg.parse_gates("Complex")
    pcg.parse_gates("Family")
    pcg.parse_2oper("Activates")
    pcg.parse_2oper("Inhibits")


#
#
#
def filter_graph():
    pcg = PSLCodeGen()
    pcg.read_graph("pathway.tab")
    genes = set()
    with open("genes.tab", "r") as f:
        for ln in f:
            ln = ln.strip()
            genes.add(ln)
    pcg.sel_graph(genes)


if __name__ == "__main__":
    filter_graph()
    parse_graph()
