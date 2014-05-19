
import random, math

#
# (patient, gene) generator
#
def pg_gen(fn, p1, p2, ng):
    fh = []
    for n in fn:
        fh.append(open(n, "w"))
    #
    gene_set = set(range(1, ng+1))
    for p in range(p1, p2+1):
        sg = random.randint(1, ng)
        for g in random.sample(gene_set, sg):
            f = fh[random.randint(0, 1)]
            v = random.random()
            f.write("{0},{1},{2:.3f}\n".format(p, g, v))
    #
    for f in fh:
        f.close()


#
# (patient, label) generator
#
def pl_gen(p1, p2):
    f = open("PatientLabel.csv", "w")
    for p in range(p1, p2+1):
        v = random.random()
        f.write("{0},{1}\n".format(p, v))


#
# gene rule generator
#
def gr_gen(ng):
    fn = ["Activates.csv", "Inhibits.csv", "Similar.csv"]
    fh = []
    for n in fn:
        fh.append(open(n, "w"))
    #
    rule_set = set()
    gene_set = set(range(1, ng+1))
    for p in range(1, random.randint(int(math.sqrt(ng)), ng*ng)):
        while True:
            (g1, g2) = random.sample(gene_set, 2)
            if (g1, g2) not in rule_set:
                break
        rule_set.add((g1, g2))
        f = fh[random.randint(0, 2)]
        f.write("{0},{1}\n".format(g1, g2))
    #
    for f in fh:
        f.close()


#
#
#
if __name__ == '__main__':
    pg_gen(["ExpUp.csv", "ExpDown.csv"], 11, 15, 9)
    pg_gen(["MutPlus.csv", "MutMinus.csv"], 11, 15, 9)
    gr_gen(9)
    #
    pl_gen(11, 15)
