

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
        rel = {"complex":"&", "family":"|"} [e_type]
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
    # generate PSL code & data for composite entities: complexes / families
    # act like AND/OR gates w/ various number of operands
    #
    def parse_gates(self, e_type="complex"):
        #
        g = self.Graph
        rel = {"complex":"component>", "family":"member>"} [e_type]
        # file handles (each file stores expressions w/ a specific number of operands)
        d_fh = {}
        # generate rules for entities
        edgeTypes = networkx.get_edge_attributes(g, "edge")
        for n in g.nodes():
            if n.endswith("({0})".format(e_type)):
                #
                if len(g.successors(n)) > 0 and len(g.predecessors(n)) > 0:
                    # select only component> predecessors
                    l_pc = [p for p in g.predecessors(n) if edgeTypes[(p, n)] == rel]
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
                    fh.write("{0}\t{1}\n".format("\t".join(self.get_node_id(p) for p in l_pc), self.get_node_id(n)))
        #
        # close all
        for fh in d_fh.values():
            fh.close()
        # output Groovy code for PSL
        self.gen_code_gates(d_fh, e_type)
        #


    #
    # generate PSL code & data for 2 operand pathway relations
    #
    # e_type = "activates" (G1, G2) or "inhibits" (G1, G2)
    #
    def parse_2oper(self, e_type):
        #
        g = self.Graph
        rel = {"activates":["-a>", "-t>", "-ap>"], "inhibits":["-a|", "-t|", "-ap|"]} [e_type]
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
#
#
if __name__ == '__main__':
    pcg = PSLCodeGen()
    pcg.read_graph("ucscSuperV3.0p2_pathway.tab")
    pcg.parse_gates("complex")
    pcg.parse_gates("family")
    pcg.parse_2oper("activates")
    pcg.parse_2oper("inhibits")
