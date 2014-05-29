package edu.ucsc.cs.utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import edu.umd.cs.psl.database.Database;
import edu.umd.cs.psl.evaluation.statistics.filter.AtomFilter;
import edu.umd.cs.psl.evaluation.statistics.filter.MaxValueFilter;
import edu.umd.cs.psl.model.argument.GroundTerm;
import edu.umd.cs.psl.model.atom.GroundAtom;
import edu.umd.cs.psl.model.atom.ObservedAtom;
import edu.umd.cs.psl.model.predicate.Predicate;
import edu.umd.cs.psl.util.database.Queries;

public class Dumper {
	
	private final Database db;
	private Predicate p;
    private String fn;

	
	public Dumper(Database db, Predicate p, String fn) {
		this.db = db;
		this.p = p;
        this.fn = fn;
	}
	
	public void outputToFile(){
		BufferedWriter writer = null;
		String dir = "./";
		String outFile = p.toString() + this.fn + ".csv";
		try {
			writer = new BufferedWriter(new FileWriter(outFile));
			
			for (GroundAtom atom : Queries.getAllAtoms(db, p)){
				GroundTerm[] terms = atom.getArguments();
				writer.append(terms[0] + "," + atom.getValue() + "\n");
                writer.flush();
			}
			writer.close();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
