
#
# convert gene expression, mutation data etc to PSL input format
#


#
# return list of top-varying genes
#
top.var = function(x, N=100) {
    vv = apply(x, 1, var)
    vv = sort(vv, decreasing=TRUE)
    return(names(vv)[1:N])
}


#
#
#
scale.01 = function(v) {
    w = (v - min(v)) / max(v)
    return(w)
}


write.active = function(samples, genes, fn) {
    mx = NULL
    for(s in samples) {
        mx = rbind(mx, cbind(genes, s))
    }
    write.table(mx, fn, sep="\t", quote=F, col.names=F, row.names=F)
}


#
#
#
format.gexp = function(x, i.train, i.test, odir=".", sd.f=0.5) {
    stopifnot(length(intersect(i.train, i.test)) == 0)

    # features are always computed relative to training
    x.train = x[, i.train]
    v.m = apply(x.train, 1, mean)
    v.sd = apply(x.train, 1, sd)
    
    #
    # train
    #
    train.up = NULL
    train.down = NULL
    train.genes = NULL
    for(i in i.train) {
        v.x = x[, i]
        z = abs(v.x) / v.sd
        cn = colnames(x)[i]
        # up
        sel = which(v.x >= v.m + v.sd * sd.f)
        v.up = cbind(rownames(x)[sel], rep(cn, length(sel)), z[sel])
        train.genes = union(train.genes, rownames(x)[sel])
        # down
        sel = which(v.x <= v.m - v.sd * sd.f)
        v.down = cbind(rownames(x)[sel], rep(cn, length(sel)), z[sel])
        train.genes = union(train.genes, rownames(x)[sel])
        # update
        train.up = rbind(train.up, v.up)
        train.down = rbind(train.down, v.down)
    }
    # scale
    train.up[, 3] = scale.01(as.numeric(train.up[, 3]))
    train.down[, 3] = scale.01(as.numeric(train.down[, 3]))
    # write Train
    fn = sprintf("%s/Train/ExpUp.tab", odir)
    write.table(train.up, fn, sep="\t", quote=F, col.names=F, row.names=F)
    fn = sprintf("%s/Train/ExpDown.tab", odir)
    write.table(train.down, fn, sep="\t", quote=F, col.names=F, row.names=F)

    #
    # test
    #
    test.up = NULL
    test.down = NULL
    test.genes = NULL
    for(i in i.test) {
        v.x = x[, i]
        z = abs(v.x) / v.sd
        cn = colnames(x)[i]
        # up
        sel = which(v.x >= v.m + v.sd * sd.f)
        v.up = cbind(rownames(x)[sel], rep(cn, length(sel)), z[sel])
        test.genes = union(test.genes, rownames(x)[sel])
        # down
        sel = which(v.x <= v.m - v.sd * sd.f)
        v.down = cbind(rownames(x)[sel], rep(cn, length(sel)), z[sel])
        test.genes = union(test.genes, rownames(x)[sel])
        # update
        test.up = rbind(test.up, v.up)
        test.down = rbind(test.down, v.down)
    }
    # scale
    test.up[, 3] = scale.01(as.numeric(test.up[, 3]))
    test.down[, 3] = scale.01(as.numeric(test.down[, 3]))
    # write Test
    fn = sprintf("%s/Test/ExpUp.tab", odir)
    write.table(test.up, fn, sep="\t", quote=F, col.names=F, row.names=F)
    fn = sprintf("%s/Test/ExpDown.tab", odir)
    write.table(test.down, fn, sep="\t", quote=F, col.names=F, row.names=F)
}


#
#
#
label.train = function(v.gleason, i.train, odir=".", fsplit=0.2) {
    # Target split
    i.target = sample(i.train, fsplit*length(i.train))
    i.nontar = setdiff(i.train, i.target)
    
    #
    # Target
    #
    nms = names(v.gleason)[i.target]
    
    # TargetPatient
    mx = cbind(nms, 1.0)
    fn = sprintf("%s/Train/TargetPatient.tab", odir)
    write.table(mx, fn, sep="\t", quote=F, col.names=F, row.names=F)

    # BogusPatientLabel
    mx = cbind(nms, 0.5)
    fn = sprintf("%s/Train/BogusPatientLabel.tab", odir)
    write.table(mx, fn, sep="\t", quote=F, col.names=F, row.names=F)
    
    # TargetPatientLabel
    mx = cbind(nms, v.gleason[i.target])
    fn = sprintf("%s/Train/TargetPatientLabel.tab", odir)
    write.table(mx, fn, sep="\t", quote=F, col.names=F, row.names=F)


    #
    # Non-Target
    #
    nms = names(v.gleason)[i.nontar]

    # PatientLabel.
    mx = cbind(nms, v.gleason[i.nontar])
    fn = sprintf("%s/Train/PatientLabel.tab", odir)
    write.table(mx, fn, sep="\t", quote=F, col.names=F, row.names=F)
}



#
#
#
label.test = function(v.gleason, i.test, odir=".") {
    #
    # Target: ALL
    #
    nms = names(v.gleason)[i.test]
    
    # TargetPatient
    mx = cbind(nms, 1.0)
    fn = sprintf("%s/Test/TargetPatient.tab", odir)
    write.table(mx, fn, sep="\t", quote=F, col.names=F, row.names=F)

    # BogusPatientLabel
    mx = cbind(nms, 0.5)
    fn = sprintf("%s/Test/BogusPatientLabel.tab", odir)
    write.table(mx, fn, sep="\t", quote=F, col.names=F, row.names=F)
    
    # PatientLabel
    mx = cbind(nms, v.gleason[i.test])
    fn = sprintf("%s/Test/PatientLabel.tab", odir)
    write.table(mx, fn, sep="\t", quote=F, col.names=F, row.names=F)
}


#
#
#
label.gene = function(x, v.fn=c("Train/Activates.tab",
    fn.fam="Train/Family.tab",
    fn.comp="Train/Complex.tab",
    fn.inhib="Train/Inhibits.tab")
    ) {
    genes = rownames(x)
    for(fn in v.fn) {
        df.fn = read.delim(fn, header=F, as.is=T, check.names=F)
        genes = union(genes, union(df.fn$V1, df.fn$V2))
    }
    #
    write.table(genes, "Train/GeneLabel.tab", sep="\t", quote=F, col.names=F, row.names=F)
    return(genes)
}



#
# Active genes
#
active.genes = function(x, genes, i.train, i.test, odir=".") {
    # train
    samples = colnames(x)[i.train]
    fn = sprintf("Train/Active.tab", odir)
    write.active(samples, genes, fn)
    # test
    samples = colnames(x)[i.test]
    fn = sprintf("Test/Active.tab", odir)
    write.active(samples, genes, fn)
}



#
# mutations
#
mutations = function(fn.maf, x, i.train, i.test, odir=".") {
    df.maf = read.delim(fn.maf, header=T, as.is=T, check.names=F, sep="\t", quote="")
    sel = which(df.maf$Variant_Classification %in% c("Frame_Shift_Del", "Frame_Shift_Ins", "Missense_Mutation", "Nonsense_Mutation"))
    maf = df.maf[sel, c("Hugo_Symbol", "Variant_Classification", "Variant_Type", "Tumor_Sample_Barcode")]
    maf$Sample = substr(maf$Tumor_Sample_Barcode, 1, 15)
    # View(maf)
    samples = colnames(x)
    #
    # train
    #
    m2 = maf[maf$Sample %in% samples[i.train], ]
    # View(m2)
    mx = cbind(m2$Hugo_Symbol, m2$Sample, 1.0)
    key = paste(mx[, 1], mx[, 2], sep="::")
    mx = mx[!duplicated(key), ]
    fn = sprintf("Train/MutMinus.tab", odir)
    write.table(mx, fn, sep="\t", quote=F, col.names=F, row.names=F)
    #
    # test
    #
    m2 = maf[maf$Sample %in% samples[i.test], ]
    # View(m2)
    mx = cbind(m2$Hugo_Symbol, m2$Sample, 1.0)
    key = paste(mx[, 1], mx[, 2], sep="::")
    mx = mx[!duplicated(key), ]
    fn = sprintf("Test/MutMinus.tab", odir)
    write.table(mx, fn, sep="\t", quote=F, col.names=F, row.names=F)
}

