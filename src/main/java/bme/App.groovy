package bme

import java.util.Set;
import edu.umd.cs.bachuai13.util.DataOutputter;
import edu.umd.cs.bachuai13.util.FoldUtils;
import edu.umd.cs.bachuai13.util.GroundingWrapper;
import edu.umd.cs.psl.application.inference.MPEInference
import edu.umd.cs.psl.application.inference.LazyMPEInference;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.LazyMaxLikelihoodMPE;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxPseudoLikelihood
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin.NormScalingType
import edu.umd.cs.psl.application.learning.weight.random.FirstOrderMetropolisRandOM
import edu.umd.cs.psl.application.learning.weight.random.HardEMRandOM
import edu.umd.cs.psl.application.learning.weight.em.HardEM
import edu.umd.cs.psl.config.*
import edu.umd.cs.psl.core.*
import edu.umd.cs.psl.core.inference.*
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database
import edu.umd.cs.psl.database.DatabasePopulator
import edu.umd.cs.psl.database.DatabaseQuery
import edu.umd.cs.psl.database.Partition
import edu.umd.cs.psl.database.ResultList
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.evaluation.result.*
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionComparator
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionStatistics
import edu.umd.cs.psl.evaluation.statistics.filter.MaxValueFilter
import edu.umd.cs.psl.groovy.*
import edu.umd.cs.psl.model.Model
import edu.umd.cs.psl.model.argument.ArgumentType
import edu.umd.cs.psl.model.argument.GroundTerm
import edu.umd.cs.psl.model.argument.UniqueID
import edu.umd.cs.psl.model.argument.Variable
import edu.umd.cs.psl.model.atom.GroundAtom
import edu.umd.cs.psl.model.atom.QueryAtom
import edu.umd.cs.psl.model.atom.RandomVariableAtom
import edu.umd.cs.psl.model.kernel.CompatibilityKernel
import edu.umd.cs.psl.model.parameters.PositiveWeight
import edu.umd.cs.psl.model.parameters.Weight
import edu.umd.cs.psl.ui.loading.*
import edu.umd.cs.psl.util.database.Queries
import edu.ucsc.cs.utils.Evaluator;

import edu.umd.cs.psl.evaluation.statistics.RankingScore
import edu.umd.cs.psl.evaluation.statistics.SimpleRankingComparator


//
dataSet = "bme"
ConfigManager cm = ConfigManager.getManager()
ConfigBundle cb = cm.getBundle(dataSet)

def defaultPath = System.getProperty("java.io.tmpdir")
//String dbPath = cb.getString("dbPath", defaultPath + File.separator + "psl-" + dataSet)
String dbPath = cb.getString("dbPath", defaultPath + File.separator + dataSet)
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbPath, true), cb)

PSLModel model = new PSLModel(this, data)

/*
 * Label predicates
 */

// PatientLabel(P) -- observed var.
model.add predicate: "PatientLabel" , types:[ArgumentType.UniqueID]
// GeneLabel(G) -- latent var.
model.add predicate: "GeneLabel" , types:[ArgumentType.UniqueID]
// TargetPatientLabel(P) -- target var.
model.add predicate: "TargetPatientLabel" , types:[ArgumentType.UniqueID]

/*
 * Evidence (observed) predicates
 */
model.add predicate: "ExpUp" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "ExpDown" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "MutPlus" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID]  // GOF
model.add predicate: "MutMinus" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID]  // LOF

// Active(G, P) -- latent
model.add predicate: "Active" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID]

// Activates(G1, G2) -- observed
model.add predicate: "Activates" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID]

// Inhibits(G1, G2) -- observed
model.add predicate: "Inhibits" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID]

// Similar(G1, G2) -- observed
model.add predicate: "Similar" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID]

// mutual exclusivity - no need to enforce these, they are all observed
// model.add rule : ExpUp(G, P) + ExpDown(G, P) <= 1.0
// model.add rule : MutPlus(G, P) + MutMinus(G, P) <= 1.0

// rules to infer gene activity from evidence
model.add rule : ExpUp(G, P) >> Active(G, P), weight : 1
model.add rule : ExpDown(G, P) >> ~Active(G, P), weight : 1
model.add rule : MutPlus(G,P) >> ~Active(G, P), weight : 1
model.add rule : MutMinus(G, P) >> Active(G, P), weight : 1

// rules to relate gene activities via gene interaction network
model.add rule : (Inhibits(G1, G2) & Active(G1, P)) >> ~Active(G2, P), weight : 1
model.add rule : (Activates(G1, G2) & Active(G1, P)) >> Active(G2, P), weight : 1
model.add rule : (Similar(G1, G2) & GeneLabel(G1)) >> GeneLabel(G2), weight : 1

// label inference
model.add rule : (PatientLabel(P) & Active(G, P)) >> GeneLabel(G), weight : 1
model.add rule : (GeneLabel(G) & Active(G, P)) >> TargetPatientLabel(P), weight : 1

// BEGIN: COMPLEX_001
model.add predicate: "COMPLEX_001", types:[ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (COMPLEX_001(G001, G_OUT) & (Active(G001, P))) >> Active(G_OUT, P), weight : 1
// END: COMPLEX_001

// BEGIN: COMPLEX_002
model.add predicate: "COMPLEX_002", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (COMPLEX_002(G001, G002, G_OUT) & (Active(G001, P) & Active(G002, P))) >> Active(G_OUT, P), weight : 1
// END: COMPLEX_002

// BEGIN: COMPLEX_003
model.add predicate: "COMPLEX_003", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (COMPLEX_003(G001, G002, G003, G_OUT) & (Active(G001, P) & Active(G002, P) & Active(G003, P))) >> Active(G_OUT, P), weight : 1
// END: COMPLEX_003

// BEGIN: COMPLEX_004
model.add predicate: "COMPLEX_004", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (COMPLEX_004(G001, G002, G003, G004, G_OUT) & (Active(G001, P) & Active(G002, P) & Active(G003, P) & Active(G004, P))) >> Active(G_OUT, P), weight : 1
// END: COMPLEX_004

// BEGIN: COMPLEX_005
model.add predicate: "COMPLEX_005", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (COMPLEX_005(G001, G002, G003, G004, G005, G_OUT) & (Active(G001, P) & Active(G002, P) & Active(G003, P) & Active(G004, P) & Active(G005, P))) >> Active(G_OUT, P), weight : 1
// END: COMPLEX_005

// BEGIN: COMPLEX_006
model.add predicate: "COMPLEX_006", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (COMPLEX_006(G001, G002, G003, G004, G005, G006, G_OUT) & (Active(G001, P) & Active(G002, P) & Active(G003, P) & Active(G004, P) & Active(G005, P) & Active(G006, P))) >> Active(G_OUT, P), weight : 1
// END: COMPLEX_006

// BEGIN: COMPLEX_007
model.add predicate: "COMPLEX_007", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (COMPLEX_007(G001, G002, G003, G004, G005, G006, G007, G_OUT) & (Active(G001, P) & Active(G002, P) & Active(G003, P) & Active(G004, P) & Active(G005, P) & Active(G006, P) & Active(G007, P))) >> Active(G_OUT, P), weight : 1
// END: COMPLEX_007

// BEGIN: COMPLEX_008
model.add predicate: "COMPLEX_008", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (COMPLEX_008(G001, G002, G003, G004, G005, G006, G007, G008, G_OUT) & (Active(G001, P) & Active(G002, P) & Active(G003, P) & Active(G004, P) & Active(G005, P) & Active(G006, P) & Active(G007, P) & Active(G008, P))) >> Active(G_OUT, P), weight : 1
// END: COMPLEX_008

// BEGIN: COMPLEX_009
model.add predicate: "COMPLEX_009", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (COMPLEX_009(G001, G002, G003, G004, G005, G006, G007, G008, G009, G_OUT) & (Active(G001, P) & Active(G002, P) & Active(G003, P) & Active(G004, P) & Active(G005, P) & Active(G006, P) & Active(G007, P) & Active(G008, P) & Active(G009, P))) >> Active(G_OUT, P), weight : 1
// END: COMPLEX_009

// BEGIN: COMPLEX_010
model.add predicate: "COMPLEX_010", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (COMPLEX_010(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G_OUT) & (Active(G001, P) & Active(G002, P) & Active(G003, P) & Active(G004, P) & Active(G005, P) & Active(G006, P) & Active(G007, P) & Active(G008, P) & Active(G009, P) & Active(G010, P))) >> Active(G_OUT, P), weight : 1
// END: COMPLEX_010

// BEGIN: COMPLEX_011
model.add predicate: "COMPLEX_011", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (COMPLEX_011(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G_OUT) & (Active(G001, P) & Active(G002, P) & Active(G003, P) & Active(G004, P) & Active(G005, P) & Active(G006, P) & Active(G007, P) & Active(G008, P) & Active(G009, P) & Active(G010, P) & Active(G011, P))) >> Active(G_OUT, P), weight : 1
// END: COMPLEX_011

// BEGIN: COMPLEX_012
model.add predicate: "COMPLEX_012", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (COMPLEX_012(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G_OUT) & (Active(G001, P) & Active(G002, P) & Active(G003, P) & Active(G004, P) & Active(G005, P) & Active(G006, P) & Active(G007, P) & Active(G008, P) & Active(G009, P) & Active(G010, P) & Active(G011, P) & Active(G012, P))) >> Active(G_OUT, P), weight : 1
// END: COMPLEX_012

// BEGIN: COMPLEX_013
model.add predicate: "COMPLEX_013", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (COMPLEX_013(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G_OUT) & (Active(G001, P) & Active(G002, P) & Active(G003, P) & Active(G004, P) & Active(G005, P) & Active(G006, P) & Active(G007, P) & Active(G008, P) & Active(G009, P) & Active(G010, P) & Active(G011, P) & Active(G012, P) & Active(G013, P))) >> Active(G_OUT, P), weight : 1
// END: COMPLEX_013

// BEGIN: COMPLEX_014
model.add predicate: "COMPLEX_014", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (COMPLEX_014(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G_OUT) & (Active(G001, P) & Active(G002, P) & Active(G003, P) & Active(G004, P) & Active(G005, P) & Active(G006, P) & Active(G007, P) & Active(G008, P) & Active(G009, P) & Active(G010, P) & Active(G011, P) & Active(G012, P) & Active(G013, P) & Active(G014, P))) >> Active(G_OUT, P), weight : 1
// END: COMPLEX_014

// BEGIN: COMPLEX_015
model.add predicate: "COMPLEX_015", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (COMPLEX_015(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G_OUT) & (Active(G001, P) & Active(G002, P) & Active(G003, P) & Active(G004, P) & Active(G005, P) & Active(G006, P) & Active(G007, P) & Active(G008, P) & Active(G009, P) & Active(G010, P) & Active(G011, P) & Active(G012, P) & Active(G013, P) & Active(G014, P) & Active(G015, P))) >> Active(G_OUT, P), weight : 1
// END: COMPLEX_015

// BEGIN: COMPLEX_016
model.add predicate: "COMPLEX_016", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (COMPLEX_016(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G_OUT) & (Active(G001, P) & Active(G002, P) & Active(G003, P) & Active(G004, P) & Active(G005, P) & Active(G006, P) & Active(G007, P) & Active(G008, P) & Active(G009, P) & Active(G010, P) & Active(G011, P) & Active(G012, P) & Active(G013, P) & Active(G014, P) & Active(G015, P) & Active(G016, P))) >> Active(G_OUT, P), weight : 1
// END: COMPLEX_016

// BEGIN: COMPLEX_017
model.add predicate: "COMPLEX_017", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (COMPLEX_017(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G_OUT) & (Active(G001, P) & Active(G002, P) & Active(G003, P) & Active(G004, P) & Active(G005, P) & Active(G006, P) & Active(G007, P) & Active(G008, P) & Active(G009, P) & Active(G010, P) & Active(G011, P) & Active(G012, P) & Active(G013, P) & Active(G014, P) & Active(G015, P) & Active(G016, P) & Active(G017, P))) >> Active(G_OUT, P), weight : 1
// END: COMPLEX_017

// BEGIN: COMPLEX_018
model.add predicate: "COMPLEX_018", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (COMPLEX_018(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G_OUT) & (Active(G001, P) & Active(G002, P) & Active(G003, P) & Active(G004, P) & Active(G005, P) & Active(G006, P) & Active(G007, P) & Active(G008, P) & Active(G009, P) & Active(G010, P) & Active(G011, P) & Active(G012, P) & Active(G013, P) & Active(G014, P) & Active(G015, P) & Active(G016, P) & Active(G017, P) & Active(G018, P))) >> Active(G_OUT, P), weight : 1
// END: COMPLEX_018

// BEGIN: COMPLEX_019
model.add predicate: "COMPLEX_019", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (COMPLEX_019(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G_OUT) & (Active(G001, P) & Active(G002, P) & Active(G003, P) & Active(G004, P) & Active(G005, P) & Active(G006, P) & Active(G007, P) & Active(G008, P) & Active(G009, P) & Active(G010, P) & Active(G011, P) & Active(G012, P) & Active(G013, P) & Active(G014, P) & Active(G015, P) & Active(G016, P) & Active(G017, P) & Active(G018, P) & Active(G019, P))) >> Active(G_OUT, P), weight : 1
// END: COMPLEX_019

// BEGIN: COMPLEX_020
model.add predicate: "COMPLEX_020", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (COMPLEX_020(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G_OUT) & (Active(G001, P) & Active(G002, P) & Active(G003, P) & Active(G004, P) & Active(G005, P) & Active(G006, P) & Active(G007, P) & Active(G008, P) & Active(G009, P) & Active(G010, P) & Active(G011, P) & Active(G012, P) & Active(G013, P) & Active(G014, P) & Active(G015, P) & Active(G016, P) & Active(G017, P) & Active(G018, P) & Active(G019, P) & Active(G020, P))) >> Active(G_OUT, P), weight : 1
// END: COMPLEX_020

// BEGIN: COMPLEX_022
model.add predicate: "COMPLEX_022", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (COMPLEX_022(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G021, G022, G_OUT) & (Active(G001, P) & Active(G002, P) & Active(G003, P) & Active(G004, P) & Active(G005, P) & Active(G006, P) & Active(G007, P) & Active(G008, P) & Active(G009, P) & Active(G010, P) & Active(G011, P) & Active(G012, P) & Active(G013, P) & Active(G014, P) & Active(G015, P) & Active(G016, P) & Active(G017, P) & Active(G018, P) & Active(G019, P) & Active(G020, P) & Active(G021, P) & Active(G022, P))) >> Active(G_OUT, P), weight : 1
// END: COMPLEX_022

// BEGIN: COMPLEX_025
model.add predicate: "COMPLEX_025", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (COMPLEX_025(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G021, G022, G023, G024, G025, G_OUT) & (Active(G001, P) & Active(G002, P) & Active(G003, P) & Active(G004, P) & Active(G005, P) & Active(G006, P) & Active(G007, P) & Active(G008, P) & Active(G009, P) & Active(G010, P) & Active(G011, P) & Active(G012, P) & Active(G013, P) & Active(G014, P) & Active(G015, P) & Active(G016, P) & Active(G017, P) & Active(G018, P) & Active(G019, P) & Active(G020, P) & Active(G021, P) & Active(G022, P) & Active(G023, P) & Active(G024, P) & Active(G025, P))) >> Active(G_OUT, P), weight : 1
// END: COMPLEX_025

// BEGIN: COMPLEX_026
model.add predicate: "COMPLEX_026", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (COMPLEX_026(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G021, G022, G023, G024, G025, G026, G_OUT) & (Active(G001, P) & Active(G002, P) & Active(G003, P) & Active(G004, P) & Active(G005, P) & Active(G006, P) & Active(G007, P) & Active(G008, P) & Active(G009, P) & Active(G010, P) & Active(G011, P) & Active(G012, P) & Active(G013, P) & Active(G014, P) & Active(G015, P) & Active(G016, P) & Active(G017, P) & Active(G018, P) & Active(G019, P) & Active(G020, P) & Active(G021, P) & Active(G022, P) & Active(G023, P) & Active(G024, P) & Active(G025, P) & Active(G026, P))) >> Active(G_OUT, P), weight : 1
// END: COMPLEX_026

// BEGIN: COMPLEX_031
model.add predicate: "COMPLEX_031", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (COMPLEX_031(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G021, G022, G023, G024, G025, G026, G027, G028, G029, G030, G031, G_OUT) & (Active(G001, P) & Active(G002, P) & Active(G003, P) & Active(G004, P) & Active(G005, P) & Active(G006, P) & Active(G007, P) & Active(G008, P) & Active(G009, P) & Active(G010, P) & Active(G011, P) & Active(G012, P) & Active(G013, P) & Active(G014, P) & Active(G015, P) & Active(G016, P) & Active(G017, P) & Active(G018, P) & Active(G019, P) & Active(G020, P) & Active(G021, P) & Active(G022, P) & Active(G023, P) & Active(G024, P) & Active(G025, P) & Active(G026, P) & Active(G027, P) & Active(G028, P) & Active(G029, P) & Active(G030, P) & Active(G031, P))) >> Active(G_OUT, P), weight : 1
// END: COMPLEX_031

// BEGIN: COMPLEX_033
model.add predicate: "COMPLEX_033", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (COMPLEX_033(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G021, G022, G023, G024, G025, G026, G027, G028, G029, G030, G031, G032, G033, G_OUT) & (Active(G001, P) & Active(G002, P) & Active(G003, P) & Active(G004, P) & Active(G005, P) & Active(G006, P) & Active(G007, P) & Active(G008, P) & Active(G009, P) & Active(G010, P) & Active(G011, P) & Active(G012, P) & Active(G013, P) & Active(G014, P) & Active(G015, P) & Active(G016, P) & Active(G017, P) & Active(G018, P) & Active(G019, P) & Active(G020, P) & Active(G021, P) & Active(G022, P) & Active(G023, P) & Active(G024, P) & Active(G025, P) & Active(G026, P) & Active(G027, P) & Active(G028, P) & Active(G029, P) & Active(G030, P) & Active(G031, P) & Active(G032, P) & Active(G033, P))) >> Active(G_OUT, P), weight : 1
// END: COMPLEX_033

// BEGIN: COMPLEX_034
model.add predicate: "COMPLEX_034", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (COMPLEX_034(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G021, G022, G023, G024, G025, G026, G027, G028, G029, G030, G031, G032, G033, G034, G_OUT) & (Active(G001, P) & Active(G002, P) & Active(G003, P) & Active(G004, P) & Active(G005, P) & Active(G006, P) & Active(G007, P) & Active(G008, P) & Active(G009, P) & Active(G010, P) & Active(G011, P) & Active(G012, P) & Active(G013, P) & Active(G014, P) & Active(G015, P) & Active(G016, P) & Active(G017, P) & Active(G018, P) & Active(G019, P) & Active(G020, P) & Active(G021, P) & Active(G022, P) & Active(G023, P) & Active(G024, P) & Active(G025, P) & Active(G026, P) & Active(G027, P) & Active(G028, P) & Active(G029, P) & Active(G030, P) & Active(G031, P) & Active(G032, P) & Active(G033, P) & Active(G034, P))) >> Active(G_OUT, P), weight : 1
// END: COMPLEX_034

// BEGIN: COMPLEX_035
model.add predicate: "COMPLEX_035", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (COMPLEX_035(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G021, G022, G023, G024, G025, G026, G027, G028, G029, G030, G031, G032, G033, G034, G035, G_OUT) & (Active(G001, P) & Active(G002, P) & Active(G003, P) & Active(G004, P) & Active(G005, P) & Active(G006, P) & Active(G007, P) & Active(G008, P) & Active(G009, P) & Active(G010, P) & Active(G011, P) & Active(G012, P) & Active(G013, P) & Active(G014, P) & Active(G015, P) & Active(G016, P) & Active(G017, P) & Active(G018, P) & Active(G019, P) & Active(G020, P) & Active(G021, P) & Active(G022, P) & Active(G023, P) & Active(G024, P) & Active(G025, P) & Active(G026, P) & Active(G027, P) & Active(G028, P) & Active(G029, P) & Active(G030, P) & Active(G031, P) & Active(G032, P) & Active(G033, P) & Active(G034, P) & Active(G035, P))) >> Active(G_OUT, P), weight : 1
// END: COMPLEX_035

// BEGIN: COMPLEX_042
model.add predicate: "COMPLEX_042", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (COMPLEX_042(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G021, G022, G023, G024, G025, G026, G027, G028, G029, G030, G031, G032, G033, G034, G035, G036, G037, G038, G039, G040, G041, G042, G_OUT) & (Active(G001, P) & Active(G002, P) & Active(G003, P) & Active(G004, P) & Active(G005, P) & Active(G006, P) & Active(G007, P) & Active(G008, P) & Active(G009, P) & Active(G010, P) & Active(G011, P) & Active(G012, P) & Active(G013, P) & Active(G014, P) & Active(G015, P) & Active(G016, P) & Active(G017, P) & Active(G018, P) & Active(G019, P) & Active(G020, P) & Active(G021, P) & Active(G022, P) & Active(G023, P) & Active(G024, P) & Active(G025, P) & Active(G026, P) & Active(G027, P) & Active(G028, P) & Active(G029, P) & Active(G030, P) & Active(G031, P) & Active(G032, P) & Active(G033, P) & Active(G034, P) & Active(G035, P) & Active(G036, P) & Active(G037, P) & Active(G038, P) & Active(G039, P) & Active(G040, P) & Active(G041, P) & Active(G042, P))) >> Active(G_OUT, P), weight : 1
// END: COMPLEX_042

// BEGIN: COMPLEX_044
model.add predicate: "COMPLEX_044", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (COMPLEX_044(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G021, G022, G023, G024, G025, G026, G027, G028, G029, G030, G031, G032, G033, G034, G035, G036, G037, G038, G039, G040, G041, G042, G043, G044, G_OUT) & (Active(G001, P) & Active(G002, P) & Active(G003, P) & Active(G004, P) & Active(G005, P) & Active(G006, P) & Active(G007, P) & Active(G008, P) & Active(G009, P) & Active(G010, P) & Active(G011, P) & Active(G012, P) & Active(G013, P) & Active(G014, P) & Active(G015, P) & Active(G016, P) & Active(G017, P) & Active(G018, P) & Active(G019, P) & Active(G020, P) & Active(G021, P) & Active(G022, P) & Active(G023, P) & Active(G024, P) & Active(G025, P) & Active(G026, P) & Active(G027, P) & Active(G028, P) & Active(G029, P) & Active(G030, P) & Active(G031, P) & Active(G032, P) & Active(G033, P) & Active(G034, P) & Active(G035, P) & Active(G036, P) & Active(G037, P) & Active(G038, P) & Active(G039, P) & Active(G040, P) & Active(G041, P) & Active(G042, P) & Active(G043, P) & Active(G044, P))) >> Active(G_OUT, P), weight : 1
// END: COMPLEX_044

// BEGIN: COMPLEX_048
model.add predicate: "COMPLEX_048", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (COMPLEX_048(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G021, G022, G023, G024, G025, G026, G027, G028, G029, G030, G031, G032, G033, G034, G035, G036, G037, G038, G039, G040, G041, G042, G043, G044, G045, G046, G047, G048, G_OUT) & (Active(G001, P) & Active(G002, P) & Active(G003, P) & Active(G004, P) & Active(G005, P) & Active(G006, P) & Active(G007, P) & Active(G008, P) & Active(G009, P) & Active(G010, P) & Active(G011, P) & Active(G012, P) & Active(G013, P) & Active(G014, P) & Active(G015, P) & Active(G016, P) & Active(G017, P) & Active(G018, P) & Active(G019, P) & Active(G020, P) & Active(G021, P) & Active(G022, P) & Active(G023, P) & Active(G024, P) & Active(G025, P) & Active(G026, P) & Active(G027, P) & Active(G028, P) & Active(G029, P) & Active(G030, P) & Active(G031, P) & Active(G032, P) & Active(G033, P) & Active(G034, P) & Active(G035, P) & Active(G036, P) & Active(G037, P) & Active(G038, P) & Active(G039, P) & Active(G040, P) & Active(G041, P) & Active(G042, P) & Active(G043, P) & Active(G044, P) & Active(G045, P) & Active(G046, P) & Active(G047, P) & Active(G048, P))) >> Active(G_OUT, P), weight : 1
// END: COMPLEX_048

// BEGIN: COMPLEX_052
model.add predicate: "COMPLEX_052", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (COMPLEX_052(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G021, G022, G023, G024, G025, G026, G027, G028, G029, G030, G031, G032, G033, G034, G035, G036, G037, G038, G039, G040, G041, G042, G043, G044, G045, G046, G047, G048, G049, G050, G051, G052, G_OUT) & (Active(G001, P) & Active(G002, P) & Active(G003, P) & Active(G004, P) & Active(G005, P) & Active(G006, P) & Active(G007, P) & Active(G008, P) & Active(G009, P) & Active(G010, P) & Active(G011, P) & Active(G012, P) & Active(G013, P) & Active(G014, P) & Active(G015, P) & Active(G016, P) & Active(G017, P) & Active(G018, P) & Active(G019, P) & Active(G020, P) & Active(G021, P) & Active(G022, P) & Active(G023, P) & Active(G024, P) & Active(G025, P) & Active(G026, P) & Active(G027, P) & Active(G028, P) & Active(G029, P) & Active(G030, P) & Active(G031, P) & Active(G032, P) & Active(G033, P) & Active(G034, P) & Active(G035, P) & Active(G036, P) & Active(G037, P) & Active(G038, P) & Active(G039, P) & Active(G040, P) & Active(G041, P) & Active(G042, P) & Active(G043, P) & Active(G044, P) & Active(G045, P) & Active(G046, P) & Active(G047, P) & Active(G048, P) & Active(G049, P) & Active(G050, P) & Active(G051, P) & Active(G052, P))) >> Active(G_OUT, P), weight : 1
// END: COMPLEX_052

// BEGIN: COMPLEX_062
model.add predicate: "COMPLEX_062", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (COMPLEX_062(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G021, G022, G023, G024, G025, G026, G027, G028, G029, G030, G031, G032, G033, G034, G035, G036, G037, G038, G039, G040, G041, G042, G043, G044, G045, G046, G047, G048, G049, G050, G051, G052, G053, G054, G055, G056, G057, G058, G059, G060, G061, G062, G_OUT) & (Active(G001, P) & Active(G002, P) & Active(G003, P) & Active(G004, P) & Active(G005, P) & Active(G006, P) & Active(G007, P) & Active(G008, P) & Active(G009, P) & Active(G010, P) & Active(G011, P) & Active(G012, P) & Active(G013, P) & Active(G014, P) & Active(G015, P) & Active(G016, P) & Active(G017, P) & Active(G018, P) & Active(G019, P) & Active(G020, P) & Active(G021, P) & Active(G022, P) & Active(G023, P) & Active(G024, P) & Active(G025, P) & Active(G026, P) & Active(G027, P) & Active(G028, P) & Active(G029, P) & Active(G030, P) & Active(G031, P) & Active(G032, P) & Active(G033, P) & Active(G034, P) & Active(G035, P) & Active(G036, P) & Active(G037, P) & Active(G038, P) & Active(G039, P) & Active(G040, P) & Active(G041, P) & Active(G042, P) & Active(G043, P) & Active(G044, P) & Active(G045, P) & Active(G046, P) & Active(G047, P) & Active(G048, P) & Active(G049, P) & Active(G050, P) & Active(G051, P) & Active(G052, P) & Active(G053, P) & Active(G054, P) & Active(G055, P) & Active(G056, P) & Active(G057, P) & Active(G058, P) & Active(G059, P) & Active(G060, P) & Active(G061, P) & Active(G062, P))) >> Active(G_OUT, P), weight : 1
// END: COMPLEX_062



// BEGIN: FAMILY_001
model.add predicate: "FAMILY_001", types:[ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_001(G001, G_OUT) & (Active(G001, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_001

// BEGIN: FAMILY_002
model.add predicate: "FAMILY_002", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_002(G001, G002, G_OUT) & (Active(G001, P) | Active(G002, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_002

// BEGIN: FAMILY_003
model.add predicate: "FAMILY_003", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_003(G001, G002, G003, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_003

// BEGIN: FAMILY_004
model.add predicate: "FAMILY_004", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_004(G001, G002, G003, G004, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_004

// BEGIN: FAMILY_005
model.add predicate: "FAMILY_005", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_005(G001, G002, G003, G004, G005, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_005

// BEGIN: FAMILY_006
model.add predicate: "FAMILY_006", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_006(G001, G002, G003, G004, G005, G006, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_006

// BEGIN: FAMILY_007
model.add predicate: "FAMILY_007", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_007(G001, G002, G003, G004, G005, G006, G007, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_007

// BEGIN: FAMILY_008
model.add predicate: "FAMILY_008", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_008(G001, G002, G003, G004, G005, G006, G007, G008, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_008

// BEGIN: FAMILY_009
model.add predicate: "FAMILY_009", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_009(G001, G002, G003, G004, G005, G006, G007, G008, G009, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P) | Active(G009, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_009

// BEGIN: FAMILY_010
model.add predicate: "FAMILY_010", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_010(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P) | Active(G009, P) | Active(G010, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_010

// BEGIN: FAMILY_011
model.add predicate: "FAMILY_011", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_011(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P) | Active(G009, P) | Active(G010, P) | Active(G011, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_011

// BEGIN: FAMILY_012
model.add predicate: "FAMILY_012", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_012(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P) | Active(G009, P) | Active(G010, P) | Active(G011, P) | Active(G012, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_012

// BEGIN: FAMILY_013
model.add predicate: "FAMILY_013", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_013(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P) | Active(G009, P) | Active(G010, P) | Active(G011, P) | Active(G012, P) | Active(G013, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_013

// BEGIN: FAMILY_014
model.add predicate: "FAMILY_014", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_014(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P) | Active(G009, P) | Active(G010, P) | Active(G011, P) | Active(G012, P) | Active(G013, P) | Active(G014, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_014

// BEGIN: FAMILY_015
model.add predicate: "FAMILY_015", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_015(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P) | Active(G009, P) | Active(G010, P) | Active(G011, P) | Active(G012, P) | Active(G013, P) | Active(G014, P) | Active(G015, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_015

// BEGIN: FAMILY_016
model.add predicate: "FAMILY_016", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_016(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P) | Active(G009, P) | Active(G010, P) | Active(G011, P) | Active(G012, P) | Active(G013, P) | Active(G014, P) | Active(G015, P) | Active(G016, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_016

// BEGIN: FAMILY_017
model.add predicate: "FAMILY_017", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_017(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P) | Active(G009, P) | Active(G010, P) | Active(G011, P) | Active(G012, P) | Active(G013, P) | Active(G014, P) | Active(G015, P) | Active(G016, P) | Active(G017, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_017

// BEGIN: FAMILY_018
model.add predicate: "FAMILY_018", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_018(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P) | Active(G009, P) | Active(G010, P) | Active(G011, P) | Active(G012, P) | Active(G013, P) | Active(G014, P) | Active(G015, P) | Active(G016, P) | Active(G017, P) | Active(G018, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_018

// BEGIN: FAMILY_019
model.add predicate: "FAMILY_019", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_019(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P) | Active(G009, P) | Active(G010, P) | Active(G011, P) | Active(G012, P) | Active(G013, P) | Active(G014, P) | Active(G015, P) | Active(G016, P) | Active(G017, P) | Active(G018, P) | Active(G019, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_019

// BEGIN: FAMILY_020
model.add predicate: "FAMILY_020", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_020(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P) | Active(G009, P) | Active(G010, P) | Active(G011, P) | Active(G012, P) | Active(G013, P) | Active(G014, P) | Active(G015, P) | Active(G016, P) | Active(G017, P) | Active(G018, P) | Active(G019, P) | Active(G020, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_020

// BEGIN: FAMILY_021
model.add predicate: "FAMILY_021", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_021(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G021, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P) | Active(G009, P) | Active(G010, P) | Active(G011, P) | Active(G012, P) | Active(G013, P) | Active(G014, P) | Active(G015, P) | Active(G016, P) | Active(G017, P) | Active(G018, P) | Active(G019, P) | Active(G020, P) | Active(G021, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_021

// BEGIN: FAMILY_024
model.add predicate: "FAMILY_024", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_024(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G021, G022, G023, G024, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P) | Active(G009, P) | Active(G010, P) | Active(G011, P) | Active(G012, P) | Active(G013, P) | Active(G014, P) | Active(G015, P) | Active(G016, P) | Active(G017, P) | Active(G018, P) | Active(G019, P) | Active(G020, P) | Active(G021, P) | Active(G022, P) | Active(G023, P) | Active(G024, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_024

// BEGIN: FAMILY_025
model.add predicate: "FAMILY_025", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_025(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G021, G022, G023, G024, G025, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P) | Active(G009, P) | Active(G010, P) | Active(G011, P) | Active(G012, P) | Active(G013, P) | Active(G014, P) | Active(G015, P) | Active(G016, P) | Active(G017, P) | Active(G018, P) | Active(G019, P) | Active(G020, P) | Active(G021, P) | Active(G022, P) | Active(G023, P) | Active(G024, P) | Active(G025, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_025

// BEGIN: FAMILY_029
model.add predicate: "FAMILY_029", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_029(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G021, G022, G023, G024, G025, G026, G027, G028, G029, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P) | Active(G009, P) | Active(G010, P) | Active(G011, P) | Active(G012, P) | Active(G013, P) | Active(G014, P) | Active(G015, P) | Active(G016, P) | Active(G017, P) | Active(G018, P) | Active(G019, P) | Active(G020, P) | Active(G021, P) | Active(G022, P) | Active(G023, P) | Active(G024, P) | Active(G025, P) | Active(G026, P) | Active(G027, P) | Active(G028, P) | Active(G029, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_029

// BEGIN: FAMILY_030
model.add predicate: "FAMILY_030", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_030(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G021, G022, G023, G024, G025, G026, G027, G028, G029, G030, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P) | Active(G009, P) | Active(G010, P) | Active(G011, P) | Active(G012, P) | Active(G013, P) | Active(G014, P) | Active(G015, P) | Active(G016, P) | Active(G017, P) | Active(G018, P) | Active(G019, P) | Active(G020, P) | Active(G021, P) | Active(G022, P) | Active(G023, P) | Active(G024, P) | Active(G025, P) | Active(G026, P) | Active(G027, P) | Active(G028, P) | Active(G029, P) | Active(G030, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_030

// BEGIN: FAMILY_031
model.add predicate: "FAMILY_031", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_031(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G021, G022, G023, G024, G025, G026, G027, G028, G029, G030, G031, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P) | Active(G009, P) | Active(G010, P) | Active(G011, P) | Active(G012, P) | Active(G013, P) | Active(G014, P) | Active(G015, P) | Active(G016, P) | Active(G017, P) | Active(G018, P) | Active(G019, P) | Active(G020, P) | Active(G021, P) | Active(G022, P) | Active(G023, P) | Active(G024, P) | Active(G025, P) | Active(G026, P) | Active(G027, P) | Active(G028, P) | Active(G029, P) | Active(G030, P) | Active(G031, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_031

// BEGIN: FAMILY_032
model.add predicate: "FAMILY_032", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_032(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G021, G022, G023, G024, G025, G026, G027, G028, G029, G030, G031, G032, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P) | Active(G009, P) | Active(G010, P) | Active(G011, P) | Active(G012, P) | Active(G013, P) | Active(G014, P) | Active(G015, P) | Active(G016, P) | Active(G017, P) | Active(G018, P) | Active(G019, P) | Active(G020, P) | Active(G021, P) | Active(G022, P) | Active(G023, P) | Active(G024, P) | Active(G025, P) | Active(G026, P) | Active(G027, P) | Active(G028, P) | Active(G029, P) | Active(G030, P) | Active(G031, P) | Active(G032, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_032

// BEGIN: FAMILY_033
model.add predicate: "FAMILY_033", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_033(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G021, G022, G023, G024, G025, G026, G027, G028, G029, G030, G031, G032, G033, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P) | Active(G009, P) | Active(G010, P) | Active(G011, P) | Active(G012, P) | Active(G013, P) | Active(G014, P) | Active(G015, P) | Active(G016, P) | Active(G017, P) | Active(G018, P) | Active(G019, P) | Active(G020, P) | Active(G021, P) | Active(G022, P) | Active(G023, P) | Active(G024, P) | Active(G025, P) | Active(G026, P) | Active(G027, P) | Active(G028, P) | Active(G029, P) | Active(G030, P) | Active(G031, P) | Active(G032, P) | Active(G033, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_033

// BEGIN: FAMILY_034
model.add predicate: "FAMILY_034", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_034(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G021, G022, G023, G024, G025, G026, G027, G028, G029, G030, G031, G032, G033, G034, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P) | Active(G009, P) | Active(G010, P) | Active(G011, P) | Active(G012, P) | Active(G013, P) | Active(G014, P) | Active(G015, P) | Active(G016, P) | Active(G017, P) | Active(G018, P) | Active(G019, P) | Active(G020, P) | Active(G021, P) | Active(G022, P) | Active(G023, P) | Active(G024, P) | Active(G025, P) | Active(G026, P) | Active(G027, P) | Active(G028, P) | Active(G029, P) | Active(G030, P) | Active(G031, P) | Active(G032, P) | Active(G033, P) | Active(G034, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_034

// BEGIN: FAMILY_037
model.add predicate: "FAMILY_037", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_037(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G021, G022, G023, G024, G025, G026, G027, G028, G029, G030, G031, G032, G033, G034, G035, G036, G037, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P) | Active(G009, P) | Active(G010, P) | Active(G011, P) | Active(G012, P) | Active(G013, P) | Active(G014, P) | Active(G015, P) | Active(G016, P) | Active(G017, P) | Active(G018, P) | Active(G019, P) | Active(G020, P) | Active(G021, P) | Active(G022, P) | Active(G023, P) | Active(G024, P) | Active(G025, P) | Active(G026, P) | Active(G027, P) | Active(G028, P) | Active(G029, P) | Active(G030, P) | Active(G031, P) | Active(G032, P) | Active(G033, P) | Active(G034, P) | Active(G035, P) | Active(G036, P) | Active(G037, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_037

// BEGIN: FAMILY_038
model.add predicate: "FAMILY_038", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_038(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G021, G022, G023, G024, G025, G026, G027, G028, G029, G030, G031, G032, G033, G034, G035, G036, G037, G038, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P) | Active(G009, P) | Active(G010, P) | Active(G011, P) | Active(G012, P) | Active(G013, P) | Active(G014, P) | Active(G015, P) | Active(G016, P) | Active(G017, P) | Active(G018, P) | Active(G019, P) | Active(G020, P) | Active(G021, P) | Active(G022, P) | Active(G023, P) | Active(G024, P) | Active(G025, P) | Active(G026, P) | Active(G027, P) | Active(G028, P) | Active(G029, P) | Active(G030, P) | Active(G031, P) | Active(G032, P) | Active(G033, P) | Active(G034, P) | Active(G035, P) | Active(G036, P) | Active(G037, P) | Active(G038, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_038

// BEGIN: FAMILY_045
model.add predicate: "FAMILY_045", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_045(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G021, G022, G023, G024, G025, G026, G027, G028, G029, G030, G031, G032, G033, G034, G035, G036, G037, G038, G039, G040, G041, G042, G043, G044, G045, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P) | Active(G009, P) | Active(G010, P) | Active(G011, P) | Active(G012, P) | Active(G013, P) | Active(G014, P) | Active(G015, P) | Active(G016, P) | Active(G017, P) | Active(G018, P) | Active(G019, P) | Active(G020, P) | Active(G021, P) | Active(G022, P) | Active(G023, P) | Active(G024, P) | Active(G025, P) | Active(G026, P) | Active(G027, P) | Active(G028, P) | Active(G029, P) | Active(G030, P) | Active(G031, P) | Active(G032, P) | Active(G033, P) | Active(G034, P) | Active(G035, P) | Active(G036, P) | Active(G037, P) | Active(G038, P) | Active(G039, P) | Active(G040, P) | Active(G041, P) | Active(G042, P) | Active(G043, P) | Active(G044, P) | Active(G045, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_045

// BEGIN: FAMILY_046
model.add predicate: "FAMILY_046", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_046(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G021, G022, G023, G024, G025, G026, G027, G028, G029, G030, G031, G032, G033, G034, G035, G036, G037, G038, G039, G040, G041, G042, G043, G044, G045, G046, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P) | Active(G009, P) | Active(G010, P) | Active(G011, P) | Active(G012, P) | Active(G013, P) | Active(G014, P) | Active(G015, P) | Active(G016, P) | Active(G017, P) | Active(G018, P) | Active(G019, P) | Active(G020, P) | Active(G021, P) | Active(G022, P) | Active(G023, P) | Active(G024, P) | Active(G025, P) | Active(G026, P) | Active(G027, P) | Active(G028, P) | Active(G029, P) | Active(G030, P) | Active(G031, P) | Active(G032, P) | Active(G033, P) | Active(G034, P) | Active(G035, P) | Active(G036, P) | Active(G037, P) | Active(G038, P) | Active(G039, P) | Active(G040, P) | Active(G041, P) | Active(G042, P) | Active(G043, P) | Active(G044, P) | Active(G045, P) | Active(G046, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_046

// BEGIN: FAMILY_052
model.add predicate: "FAMILY_052", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_052(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G021, G022, G023, G024, G025, G026, G027, G028, G029, G030, G031, G032, G033, G034, G035, G036, G037, G038, G039, G040, G041, G042, G043, G044, G045, G046, G047, G048, G049, G050, G051, G052, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P) | Active(G009, P) | Active(G010, P) | Active(G011, P) | Active(G012, P) | Active(G013, P) | Active(G014, P) | Active(G015, P) | Active(G016, P) | Active(G017, P) | Active(G018, P) | Active(G019, P) | Active(G020, P) | Active(G021, P) | Active(G022, P) | Active(G023, P) | Active(G024, P) | Active(G025, P) | Active(G026, P) | Active(G027, P) | Active(G028, P) | Active(G029, P) | Active(G030, P) | Active(G031, P) | Active(G032, P) | Active(G033, P) | Active(G034, P) | Active(G035, P) | Active(G036, P) | Active(G037, P) | Active(G038, P) | Active(G039, P) | Active(G040, P) | Active(G041, P) | Active(G042, P) | Active(G043, P) | Active(G044, P) | Active(G045, P) | Active(G046, P) | Active(G047, P) | Active(G048, P) | Active(G049, P) | Active(G050, P) | Active(G051, P) | Active(G052, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_052

// BEGIN: FAMILY_054
model.add predicate: "FAMILY_054", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_054(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G021, G022, G023, G024, G025, G026, G027, G028, G029, G030, G031, G032, G033, G034, G035, G036, G037, G038, G039, G040, G041, G042, G043, G044, G045, G046, G047, G048, G049, G050, G051, G052, G053, G054, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P) | Active(G009, P) | Active(G010, P) | Active(G011, P) | Active(G012, P) | Active(G013, P) | Active(G014, P) | Active(G015, P) | Active(G016, P) | Active(G017, P) | Active(G018, P) | Active(G019, P) | Active(G020, P) | Active(G021, P) | Active(G022, P) | Active(G023, P) | Active(G024, P) | Active(G025, P) | Active(G026, P) | Active(G027, P) | Active(G028, P) | Active(G029, P) | Active(G030, P) | Active(G031, P) | Active(G032, P) | Active(G033, P) | Active(G034, P) | Active(G035, P) | Active(G036, P) | Active(G037, P) | Active(G038, P) | Active(G039, P) | Active(G040, P) | Active(G041, P) | Active(G042, P) | Active(G043, P) | Active(G044, P) | Active(G045, P) | Active(G046, P) | Active(G047, P) | Active(G048, P) | Active(G049, P) | Active(G050, P) | Active(G051, P) | Active(G052, P) | Active(G053, P) | Active(G054, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_054

// BEGIN: FAMILY_057
model.add predicate: "FAMILY_057", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_057(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G021, G022, G023, G024, G025, G026, G027, G028, G029, G030, G031, G032, G033, G034, G035, G036, G037, G038, G039, G040, G041, G042, G043, G044, G045, G046, G047, G048, G049, G050, G051, G052, G053, G054, G055, G056, G057, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P) | Active(G009, P) | Active(G010, P) | Active(G011, P) | Active(G012, P) | Active(G013, P) | Active(G014, P) | Active(G015, P) | Active(G016, P) | Active(G017, P) | Active(G018, P) | Active(G019, P) | Active(G020, P) | Active(G021, P) | Active(G022, P) | Active(G023, P) | Active(G024, P) | Active(G025, P) | Active(G026, P) | Active(G027, P) | Active(G028, P) | Active(G029, P) | Active(G030, P) | Active(G031, P) | Active(G032, P) | Active(G033, P) | Active(G034, P) | Active(G035, P) | Active(G036, P) | Active(G037, P) | Active(G038, P) | Active(G039, P) | Active(G040, P) | Active(G041, P) | Active(G042, P) | Active(G043, P) | Active(G044, P) | Active(G045, P) | Active(G046, P) | Active(G047, P) | Active(G048, P) | Active(G049, P) | Active(G050, P) | Active(G051, P) | Active(G052, P) | Active(G053, P) | Active(G054, P) | Active(G055, P) | Active(G056, P) | Active(G057, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_057

// BEGIN: FAMILY_058
model.add predicate: "FAMILY_058", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_058(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G021, G022, G023, G024, G025, G026, G027, G028, G029, G030, G031, G032, G033, G034, G035, G036, G037, G038, G039, G040, G041, G042, G043, G044, G045, G046, G047, G048, G049, G050, G051, G052, G053, G054, G055, G056, G057, G058, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P) | Active(G009, P) | Active(G010, P) | Active(G011, P) | Active(G012, P) | Active(G013, P) | Active(G014, P) | Active(G015, P) | Active(G016, P) | Active(G017, P) | Active(G018, P) | Active(G019, P) | Active(G020, P) | Active(G021, P) | Active(G022, P) | Active(G023, P) | Active(G024, P) | Active(G025, P) | Active(G026, P) | Active(G027, P) | Active(G028, P) | Active(G029, P) | Active(G030, P) | Active(G031, P) | Active(G032, P) | Active(G033, P) | Active(G034, P) | Active(G035, P) | Active(G036, P) | Active(G037, P) | Active(G038, P) | Active(G039, P) | Active(G040, P) | Active(G041, P) | Active(G042, P) | Active(G043, P) | Active(G044, P) | Active(G045, P) | Active(G046, P) | Active(G047, P) | Active(G048, P) | Active(G049, P) | Active(G050, P) | Active(G051, P) | Active(G052, P) | Active(G053, P) | Active(G054, P) | Active(G055, P) | Active(G056, P) | Active(G057, P) | Active(G058, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_058

// BEGIN: FAMILY_063
model.add predicate: "FAMILY_063", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_063(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G021, G022, G023, G024, G025, G026, G027, G028, G029, G030, G031, G032, G033, G034, G035, G036, G037, G038, G039, G040, G041, G042, G043, G044, G045, G046, G047, G048, G049, G050, G051, G052, G053, G054, G055, G056, G057, G058, G059, G060, G061, G062, G063, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P) | Active(G009, P) | Active(G010, P) | Active(G011, P) | Active(G012, P) | Active(G013, P) | Active(G014, P) | Active(G015, P) | Active(G016, P) | Active(G017, P) | Active(G018, P) | Active(G019, P) | Active(G020, P) | Active(G021, P) | Active(G022, P) | Active(G023, P) | Active(G024, P) | Active(G025, P) | Active(G026, P) | Active(G027, P) | Active(G028, P) | Active(G029, P) | Active(G030, P) | Active(G031, P) | Active(G032, P) | Active(G033, P) | Active(G034, P) | Active(G035, P) | Active(G036, P) | Active(G037, P) | Active(G038, P) | Active(G039, P) | Active(G040, P) | Active(G041, P) | Active(G042, P) | Active(G043, P) | Active(G044, P) | Active(G045, P) | Active(G046, P) | Active(G047, P) | Active(G048, P) | Active(G049, P) | Active(G050, P) | Active(G051, P) | Active(G052, P) | Active(G053, P) | Active(G054, P) | Active(G055, P) | Active(G056, P) | Active(G057, P) | Active(G058, P) | Active(G059, P) | Active(G060, P) | Active(G061, P) | Active(G062, P) | Active(G063, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_063

// BEGIN: FAMILY_193
model.add predicate: "FAMILY_193", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_193(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G021, G022, G023, G024, G025, G026, G027, G028, G029, G030, G031, G032, G033, G034, G035, G036, G037, G038, G039, G040, G041, G042, G043, G044, G045, G046, G047, G048, G049, G050, G051, G052, G053, G054, G055, G056, G057, G058, G059, G060, G061, G062, G063, G064, G065, G066, G067, G068, G069, G070, G071, G072, G073, G074, G075, G076, G077, G078, G079, G080, G081, G082, G083, G084, G085, G086, G087, G088, G089, G090, G091, G092, G093, G094, G095, G096, G097, G098, G099, G100, G101, G102, G103, G104, G105, G106, G107, G108, G109, G110, G111, G112, G113, G114, G115, G116, G117, G118, G119, G120, G121, G122, G123, G124, G125, G126, G127, G128, G129, G130, G131, G132, G133, G134, G135, G136, G137, G138, G139, G140, G141, G142, G143, G144, G145, G146, G147, G148, G149, G150, G151, G152, G153, G154, G155, G156, G157, G158, G159, G160, G161, G162, G163, G164, G165, G166, G167, G168, G169, G170, G171, G172, G173, G174, G175, G176, G177, G178, G179, G180, G181, G182, G183, G184, G185, G186, G187, G188, G189, G190, G191, G192, G193, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P) | Active(G009, P) | Active(G010, P) | Active(G011, P) | Active(G012, P) | Active(G013, P) | Active(G014, P) | Active(G015, P) | Active(G016, P) | Active(G017, P) | Active(G018, P) | Active(G019, P) | Active(G020, P) | Active(G021, P) | Active(G022, P) | Active(G023, P) | Active(G024, P) | Active(G025, P) | Active(G026, P) | Active(G027, P) | Active(G028, P) | Active(G029, P) | Active(G030, P) | Active(G031, P) | Active(G032, P) | Active(G033, P) | Active(G034, P) | Active(G035, P) | Active(G036, P) | Active(G037, P) | Active(G038, P) | Active(G039, P) | Active(G040, P) | Active(G041, P) | Active(G042, P) | Active(G043, P) | Active(G044, P) | Active(G045, P) | Active(G046, P) | Active(G047, P) | Active(G048, P) | Active(G049, P) | Active(G050, P) | Active(G051, P) | Active(G052, P) | Active(G053, P) | Active(G054, P) | Active(G055, P) | Active(G056, P) | Active(G057, P) | Active(G058, P) | Active(G059, P) | Active(G060, P) | Active(G061, P) | Active(G062, P) | Active(G063, P) | Active(G064, P) | Active(G065, P) | Active(G066, P) | Active(G067, P) | Active(G068, P) | Active(G069, P) | Active(G070, P) | Active(G071, P) | Active(G072, P) | Active(G073, P) | Active(G074, P) | Active(G075, P) | Active(G076, P) | Active(G077, P) | Active(G078, P) | Active(G079, P) | Active(G080, P) | Active(G081, P) | Active(G082, P) | Active(G083, P) | Active(G084, P) | Active(G085, P) | Active(G086, P) | Active(G087, P) | Active(G088, P) | Active(G089, P) | Active(G090, P) | Active(G091, P) | Active(G092, P) | Active(G093, P) | Active(G094, P) | Active(G095, P) | Active(G096, P) | Active(G097, P) | Active(G098, P) | Active(G099, P) | Active(G100, P) | Active(G101, P) | Active(G102, P) | Active(G103, P) | Active(G104, P) | Active(G105, P) | Active(G106, P) | Active(G107, P) | Active(G108, P) | Active(G109, P) | Active(G110, P) | Active(G111, P) | Active(G112, P) | Active(G113, P) | Active(G114, P) | Active(G115, P) | Active(G116, P) | Active(G117, P) | Active(G118, P) | Active(G119, P) | Active(G120, P) | Active(G121, P) | Active(G122, P) | Active(G123, P) | Active(G124, P) | Active(G125, P) | Active(G126, P) | Active(G127, P) | Active(G128, P) | Active(G129, P) | Active(G130, P) | Active(G131, P) | Active(G132, P) | Active(G133, P) | Active(G134, P) | Active(G135, P) | Active(G136, P) | Active(G137, P) | Active(G138, P) | Active(G139, P) | Active(G140, P) | Active(G141, P) | Active(G142, P) | Active(G143, P) | Active(G144, P) | Active(G145, P) | Active(G146, P) | Active(G147, P) | Active(G148, P) | Active(G149, P) | Active(G150, P) | Active(G151, P) | Active(G152, P) | Active(G153, P) | Active(G154, P) | Active(G155, P) | Active(G156, P) | Active(G157, P) | Active(G158, P) | Active(G159, P) | Active(G160, P) | Active(G161, P) | Active(G162, P) | Active(G163, P) | Active(G164, P) | Active(G165, P) | Active(G166, P) | Active(G167, P) | Active(G168, P) | Active(G169, P) | Active(G170, P) | Active(G171, P) | Active(G172, P) | Active(G173, P) | Active(G174, P) | Active(G175, P) | Active(G176, P) | Active(G177, P) | Active(G178, P) | Active(G179, P) | Active(G180, P) | Active(G181, P) | Active(G182, P) | Active(G183, P) | Active(G184, P) | Active(G185, P) | Active(G186, P) | Active(G187, P) | Active(G188, P) | Active(G189, P) | Active(G190, P) | Active(G191, P) | Active(G192, P) | Active(G193, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_193

// BEGIN: FAMILY_069
model.add predicate: "FAMILY_069", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_069(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G021, G022, G023, G024, G025, G026, G027, G028, G029, G030, G031, G032, G033, G034, G035, G036, G037, G038, G039, G040, G041, G042, G043, G044, G045, G046, G047, G048, G049, G050, G051, G052, G053, G054, G055, G056, G057, G058, G059, G060, G061, G062, G063, G064, G065, G066, G067, G068, G069, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P) | Active(G009, P) | Active(G010, P) | Active(G011, P) | Active(G012, P) | Active(G013, P) | Active(G014, P) | Active(G015, P) | Active(G016, P) | Active(G017, P) | Active(G018, P) | Active(G019, P) | Active(G020, P) | Active(G021, P) | Active(G022, P) | Active(G023, P) | Active(G024, P) | Active(G025, P) | Active(G026, P) | Active(G027, P) | Active(G028, P) | Active(G029, P) | Active(G030, P) | Active(G031, P) | Active(G032, P) | Active(G033, P) | Active(G034, P) | Active(G035, P) | Active(G036, P) | Active(G037, P) | Active(G038, P) | Active(G039, P) | Active(G040, P) | Active(G041, P) | Active(G042, P) | Active(G043, P) | Active(G044, P) | Active(G045, P) | Active(G046, P) | Active(G047, P) | Active(G048, P) | Active(G049, P) | Active(G050, P) | Active(G051, P) | Active(G052, P) | Active(G053, P) | Active(G054, P) | Active(G055, P) | Active(G056, P) | Active(G057, P) | Active(G058, P) | Active(G059, P) | Active(G060, P) | Active(G061, P) | Active(G062, P) | Active(G063, P) | Active(G064, P) | Active(G065, P) | Active(G066, P) | Active(G067, P) | Active(G068, P) | Active(G069, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_069

// BEGIN: FAMILY_083
model.add predicate: "FAMILY_083", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_083(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G021, G022, G023, G024, G025, G026, G027, G028, G029, G030, G031, G032, G033, G034, G035, G036, G037, G038, G039, G040, G041, G042, G043, G044, G045, G046, G047, G048, G049, G050, G051, G052, G053, G054, G055, G056, G057, G058, G059, G060, G061, G062, G063, G064, G065, G066, G067, G068, G069, G070, G071, G072, G073, G074, G075, G076, G077, G078, G079, G080, G081, G082, G083, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P) | Active(G009, P) | Active(G010, P) | Active(G011, P) | Active(G012, P) | Active(G013, P) | Active(G014, P) | Active(G015, P) | Active(G016, P) | Active(G017, P) | Active(G018, P) | Active(G019, P) | Active(G020, P) | Active(G021, P) | Active(G022, P) | Active(G023, P) | Active(G024, P) | Active(G025, P) | Active(G026, P) | Active(G027, P) | Active(G028, P) | Active(G029, P) | Active(G030, P) | Active(G031, P) | Active(G032, P) | Active(G033, P) | Active(G034, P) | Active(G035, P) | Active(G036, P) | Active(G037, P) | Active(G038, P) | Active(G039, P) | Active(G040, P) | Active(G041, P) | Active(G042, P) | Active(G043, P) | Active(G044, P) | Active(G045, P) | Active(G046, P) | Active(G047, P) | Active(G048, P) | Active(G049, P) | Active(G050, P) | Active(G051, P) | Active(G052, P) | Active(G053, P) | Active(G054, P) | Active(G055, P) | Active(G056, P) | Active(G057, P) | Active(G058, P) | Active(G059, P) | Active(G060, P) | Active(G061, P) | Active(G062, P) | Active(G063, P) | Active(G064, P) | Active(G065, P) | Active(G066, P) | Active(G067, P) | Active(G068, P) | Active(G069, P) | Active(G070, P) | Active(G071, P) | Active(G072, P) | Active(G073, P) | Active(G074, P) | Active(G075, P) | Active(G076, P) | Active(G077, P) | Active(G078, P) | Active(G079, P) | Active(G080, P) | Active(G081, P) | Active(G082, P) | Active(G083, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_083

// BEGIN: FAMILY_374
model.add predicate: "FAMILY_374", types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add rule : (FAMILY_374(G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014, G015, G016, G017, G018, G019, G020, G021, G022, G023, G024, G025, G026, G027, G028, G029, G030, G031, G032, G033, G034, G035, G036, G037, G038, G039, G040, G041, G042, G043, G044, G045, G046, G047, G048, G049, G050, G051, G052, G053, G054, G055, G056, G057, G058, G059, G060, G061, G062, G063, G064, G065, G066, G067, G068, G069, G070, G071, G072, G073, G074, G075, G076, G077, G078, G079, G080, G081, G082, G083, G084, G085, G086, G087, G088, G089, G090, G091, G092, G093, G094, G095, G096, G097, G098, G099, G100, G101, G102, G103, G104, G105, G106, G107, G108, G109, G110, G111, G112, G113, G114, G115, G116, G117, G118, G119, G120, G121, G122, G123, G124, G125, G126, G127, G128, G129, G130, G131, G132, G133, G134, G135, G136, G137, G138, G139, G140, G141, G142, G143, G144, G145, G146, G147, G148, G149, G150, G151, G152, G153, G154, G155, G156, G157, G158, G159, G160, G161, G162, G163, G164, G165, G166, G167, G168, G169, G170, G171, G172, G173, G174, G175, G176, G177, G178, G179, G180, G181, G182, G183, G184, G185, G186, G187, G188, G189, G190, G191, G192, G193, G194, G195, G196, G197, G198, G199, G200, G201, G202, G203, G204, G205, G206, G207, G208, G209, G210, G211, G212, G213, G214, G215, G216, G217, G218, G219, G220, G221, G222, G223, G224, G225, G226, G227, G228, G229, G230, G231, G232, G233, G234, G235, G236, G237, G238, G239, G240, G241, G242, G243, G244, G245, G246, G247, G248, G249, G250, G251, G252, G253, G254, G255, G256, G257, G258, G259, G260, G261, G262, G263, G264, G265, G266, G267, G268, G269, G270, G271, G272, G273, G274, G275, G276, G277, G278, G279, G280, G281, G282, G283, G284, G285, G286, G287, G288, G289, G290, G291, G292, G293, G294, G295, G296, G297, G298, G299, G300, G301, G302, G303, G304, G305, G306, G307, G308, G309, G310, G311, G312, G313, G314, G315, G316, G317, G318, G319, G320, G321, G322, G323, G324, G325, G326, G327, G328, G329, G330, G331, G332, G333, G334, G335, G336, G337, G338, G339, G340, G341, G342, G343, G344, G345, G346, G347, G348, G349, G350, G351, G352, G353, G354, G355, G356, G357, G358, G359, G360, G361, G362, G363, G364, G365, G366, G367, G368, G369, G370, G371, G372, G373, G374, G_OUT) & (Active(G001, P) | Active(G002, P) | Active(G003, P) | Active(G004, P) | Active(G005, P) | Active(G006, P) | Active(G007, P) | Active(G008, P) | Active(G009, P) | Active(G010, P) | Active(G011, P) | Active(G012, P) | Active(G013, P) | Active(G014, P) | Active(G015, P) | Active(G016, P) | Active(G017, P) | Active(G018, P) | Active(G019, P) | Active(G020, P) | Active(G021, P) | Active(G022, P) | Active(G023, P) | Active(G024, P) | Active(G025, P) | Active(G026, P) | Active(G027, P) | Active(G028, P) | Active(G029, P) | Active(G030, P) | Active(G031, P) | Active(G032, P) | Active(G033, P) | Active(G034, P) | Active(G035, P) | Active(G036, P) | Active(G037, P) | Active(G038, P) | Active(G039, P) | Active(G040, P) | Active(G041, P) | Active(G042, P) | Active(G043, P) | Active(G044, P) | Active(G045, P) | Active(G046, P) | Active(G047, P) | Active(G048, P) | Active(G049, P) | Active(G050, P) | Active(G051, P) | Active(G052, P) | Active(G053, P) | Active(G054, P) | Active(G055, P) | Active(G056, P) | Active(G057, P) | Active(G058, P) | Active(G059, P) | Active(G060, P) | Active(G061, P) | Active(G062, P) | Active(G063, P) | Active(G064, P) | Active(G065, P) | Active(G066, P) | Active(G067, P) | Active(G068, P) | Active(G069, P) | Active(G070, P) | Active(G071, P) | Active(G072, P) | Active(G073, P) | Active(G074, P) | Active(G075, P) | Active(G076, P) | Active(G077, P) | Active(G078, P) | Active(G079, P) | Active(G080, P) | Active(G081, P) | Active(G082, P) | Active(G083, P) | Active(G084, P) | Active(G085, P) | Active(G086, P) | Active(G087, P) | Active(G088, P) | Active(G089, P) | Active(G090, P) | Active(G091, P) | Active(G092, P) | Active(G093, P) | Active(G094, P) | Active(G095, P) | Active(G096, P) | Active(G097, P) | Active(G098, P) | Active(G099, P) | Active(G100, P) | Active(G101, P) | Active(G102, P) | Active(G103, P) | Active(G104, P) | Active(G105, P) | Active(G106, P) | Active(G107, P) | Active(G108, P) | Active(G109, P) | Active(G110, P) | Active(G111, P) | Active(G112, P) | Active(G113, P) | Active(G114, P) | Active(G115, P) | Active(G116, P) | Active(G117, P) | Active(G118, P) | Active(G119, P) | Active(G120, P) | Active(G121, P) | Active(G122, P) | Active(G123, P) | Active(G124, P) | Active(G125, P) | Active(G126, P) | Active(G127, P) | Active(G128, P) | Active(G129, P) | Active(G130, P) | Active(G131, P) | Active(G132, P) | Active(G133, P) | Active(G134, P) | Active(G135, P) | Active(G136, P) | Active(G137, P) | Active(G138, P) | Active(G139, P) | Active(G140, P) | Active(G141, P) | Active(G142, P) | Active(G143, P) | Active(G144, P) | Active(G145, P) | Active(G146, P) | Active(G147, P) | Active(G148, P) | Active(G149, P) | Active(G150, P) | Active(G151, P) | Active(G152, P) | Active(G153, P) | Active(G154, P) | Active(G155, P) | Active(G156, P) | Active(G157, P) | Active(G158, P) | Active(G159, P) | Active(G160, P) | Active(G161, P) | Active(G162, P) | Active(G163, P) | Active(G164, P) | Active(G165, P) | Active(G166, P) | Active(G167, P) | Active(G168, P) | Active(G169, P) | Active(G170, P) | Active(G171, P) | Active(G172, P) | Active(G173, P) | Active(G174, P) | Active(G175, P) | Active(G176, P) | Active(G177, P) | Active(G178, P) | Active(G179, P) | Active(G180, P) | Active(G181, P) | Active(G182, P) | Active(G183, P) | Active(G184, P) | Active(G185, P) | Active(G186, P) | Active(G187, P) | Active(G188, P) | Active(G189, P) | Active(G190, P) | Active(G191, P) | Active(G192, P) | Active(G193, P) | Active(G194, P) | Active(G195, P) | Active(G196, P) | Active(G197, P) | Active(G198, P) | Active(G199, P) | Active(G200, P) | Active(G201, P) | Active(G202, P) | Active(G203, P) | Active(G204, P) | Active(G205, P) | Active(G206, P) | Active(G207, P) | Active(G208, P) | Active(G209, P) | Active(G210, P) | Active(G211, P) | Active(G212, P) | Active(G213, P) | Active(G214, P) | Active(G215, P) | Active(G216, P) | Active(G217, P) | Active(G218, P) | Active(G219, P) | Active(G220, P) | Active(G221, P) | Active(G222, P) | Active(G223, P) | Active(G224, P) | Active(G225, P) | Active(G226, P) | Active(G227, P) | Active(G228, P) | Active(G229, P) | Active(G230, P) | Active(G231, P) | Active(G232, P) | Active(G233, P) | Active(G234, P) | Active(G235, P) | Active(G236, P) | Active(G237, P) | Active(G238, P) | Active(G239, P) | Active(G240, P) | Active(G241, P) | Active(G242, P) | Active(G243, P) | Active(G244, P) | Active(G245, P) | Active(G246, P) | Active(G247, P) | Active(G248, P) | Active(G249, P) | Active(G250, P) | Active(G251, P) | Active(G252, P) | Active(G253, P) | Active(G254, P) | Active(G255, P) | Active(G256, P) | Active(G257, P) | Active(G258, P) | Active(G259, P) | Active(G260, P) | Active(G261, P) | Active(G262, P) | Active(G263, P) | Active(G264, P) | Active(G265, P) | Active(G266, P) | Active(G267, P) | Active(G268, P) | Active(G269, P) | Active(G270, P) | Active(G271, P) | Active(G272, P) | Active(G273, P) | Active(G274, P) | Active(G275, P) | Active(G276, P) | Active(G277, P) | Active(G278, P) | Active(G279, P) | Active(G280, P) | Active(G281, P) | Active(G282, P) | Active(G283, P) | Active(G284, P) | Active(G285, P) | Active(G286, P) | Active(G287, P) | Active(G288, P) | Active(G289, P) | Active(G290, P) | Active(G291, P) | Active(G292, P) | Active(G293, P) | Active(G294, P) | Active(G295, P) | Active(G296, P) | Active(G297, P) | Active(G298, P) | Active(G299, P) | Active(G300, P) | Active(G301, P) | Active(G302, P) | Active(G303, P) | Active(G304, P) | Active(G305, P) | Active(G306, P) | Active(G307, P) | Active(G308, P) | Active(G309, P) | Active(G310, P) | Active(G311, P) | Active(G312, P) | Active(G313, P) | Active(G314, P) | Active(G315, P) | Active(G316, P) | Active(G317, P) | Active(G318, P) | Active(G319, P) | Active(G320, P) | Active(G321, P) | Active(G322, P) | Active(G323, P) | Active(G324, P) | Active(G325, P) | Active(G326, P) | Active(G327, P) | Active(G328, P) | Active(G329, P) | Active(G330, P) | Active(G331, P) | Active(G332, P) | Active(G333, P) | Active(G334, P) | Active(G335, P) | Active(G336, P) | Active(G337, P) | Active(G338, P) | Active(G339, P) | Active(G340, P) | Active(G341, P) | Active(G342, P) | Active(G343, P) | Active(G344, P) | Active(G345, P) | Active(G346, P) | Active(G347, P) | Active(G348, P) | Active(G349, P) | Active(G350, P) | Active(G351, P) | Active(G352, P) | Active(G353, P) | Active(G354, P) | Active(G355, P) | Active(G356, P) | Active(G357, P) | Active(G358, P) | Active(G359, P) | Active(G360, P) | Active(G361, P) | Active(G362, P) | Active(G363, P) | Active(G364, P) | Active(G365, P) | Active(G366, P) | Active(G367, P) | Active(G368, P) | Active(G369, P) | Active(G370, P) | Active(G371, P) | Active(G372, P) | Active(G373, P) | Active(G374, P))) >> Active(G_OUT, P), weight : 1
// END: FAMILY_374


/*
 * Inserting data into the data store
 */
//fold = 1

//foldStr = "fold" + String.valueOf(fold) + java.io.File.separator;

/* training partitions */
Partition observed_tr = new Partition(0);
Partition predict_tr = new Partition(1);
Partition truth_tr = new Partition(2);
Partition dummy_tr = new Partition(3);

/*testing partitions */
Partition observed_te = new Partition(4);
Partition predict_te = new Partition(5);
Partition dummy_te = new Partition(6);

/*separate partitions for the gold standard truth for testing */
Partition PatientLabelTruth = new Partition(7);


//def dir = 'data'+java.io.File.separator+ foldStr + 'train'+java.io.File.separator;
def traindir = 'data' + java.io.File.separator + 'train' + java.io.File.separator;

/*
 * Observed data partition
 * 
 */

inserter = data.getInserter(ExpUp, observed_tr)
InserterUtils.loadDelimitedDataTruth(inserter, traindir + "ExpUp.csv", ",");

inserter = data.getInserter(ExpDown, observed_tr)
InserterUtils.loadDelimitedDataTruth(inserter, traindir + "ExpDown.csv", ",");

inserter = data.getInserter(MutPlus, observed_tr)
InserterUtils.loadDelimitedDataTruth(inserter, traindir + "MutPlus.csv", ",");

inserter = data.getInserter(MutMinus, observed_tr)
InserterUtils.loadDelimitedDataTruth(inserter, traindir + "MutMinus.csv", ",");

inserter = data.getInserter(PatientLabel, observed_tr)
InserterUtils.loadDelimitedDataTruth(inserter, traindir + "PatientLabel.csv", ",");

/*
 * Ground truth for training data for weight learning
 */

inserter = data.getInserter(TargetPatientLabel, truth_tr)
InserterUtils.loadDelimitedDataTruth(inserter, traindir + "TargetPatientLabel.csv", ",")


/*
 * Used later on to populate training DB with all possible interactions
 * ??? predicates on latents ???
 */

inserter = data.getInserter(Activates, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "activates.tab", "\t")

inserter = data.getInserter(Inhibits, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "inhibits.tab", "\t")

inserter = data.getInserter(Similar, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "Similar.csv", ",")

inserter = data.getInserter(TargetPatientLabel, dummy_tr)
InserterUtils.loadDelimitedDataTruth(inserter, traindir + "BogusPatientLabel.csv", ",")

// BEGIN: COMPLEX_001
inserter = data.getInserter(COMPLEX_001, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "complex_001.tab", "\t")
// END: COMPLEX_001

// BEGIN: COMPLEX_002
inserter = data.getInserter(COMPLEX_002, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "complex_002.tab", "\t")
// END: COMPLEX_002

// BEGIN: COMPLEX_003
inserter = data.getInserter(COMPLEX_003, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "complex_003.tab", "\t")
// END: COMPLEX_003

// BEGIN: COMPLEX_004
inserter = data.getInserter(COMPLEX_004, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "complex_004.tab", "\t")
// END: COMPLEX_004

// BEGIN: COMPLEX_005
inserter = data.getInserter(COMPLEX_005, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "complex_005.tab", "\t")
// END: COMPLEX_005

// BEGIN: COMPLEX_006
inserter = data.getInserter(COMPLEX_006, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "complex_006.tab", "\t")
// END: COMPLEX_006

// BEGIN: COMPLEX_007
inserter = data.getInserter(COMPLEX_007, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "complex_007.tab", "\t")
// END: COMPLEX_007

// BEGIN: COMPLEX_008
inserter = data.getInserter(COMPLEX_008, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "complex_008.tab", "\t")
// END: COMPLEX_008

// BEGIN: COMPLEX_009
inserter = data.getInserter(COMPLEX_009, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "complex_009.tab", "\t")
// END: COMPLEX_009

// BEGIN: COMPLEX_010
inserter = data.getInserter(COMPLEX_010, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "complex_010.tab", "\t")
// END: COMPLEX_010

// BEGIN: COMPLEX_011
inserter = data.getInserter(COMPLEX_011, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "complex_011.tab", "\t")
// END: COMPLEX_011

// BEGIN: COMPLEX_012
inserter = data.getInserter(COMPLEX_012, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "complex_012.tab", "\t")
// END: COMPLEX_012

// BEGIN: COMPLEX_013
inserter = data.getInserter(COMPLEX_013, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "complex_013.tab", "\t")
// END: COMPLEX_013

// BEGIN: COMPLEX_014
inserter = data.getInserter(COMPLEX_014, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "complex_014.tab", "\t")
// END: COMPLEX_014

// BEGIN: COMPLEX_015
inserter = data.getInserter(COMPLEX_015, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "complex_015.tab", "\t")
// END: COMPLEX_015

// BEGIN: COMPLEX_016
inserter = data.getInserter(COMPLEX_016, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "complex_016.tab", "\t")
// END: COMPLEX_016

// BEGIN: COMPLEX_017
inserter = data.getInserter(COMPLEX_017, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "complex_017.tab", "\t")
// END: COMPLEX_017

// BEGIN: COMPLEX_018
inserter = data.getInserter(COMPLEX_018, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "complex_018.tab", "\t")
// END: COMPLEX_018

// BEGIN: COMPLEX_019
inserter = data.getInserter(COMPLEX_019, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "complex_019.tab", "\t")
// END: COMPLEX_019

// BEGIN: COMPLEX_020
inserter = data.getInserter(COMPLEX_020, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "complex_020.tab", "\t")
// END: COMPLEX_020

// BEGIN: COMPLEX_022
inserter = data.getInserter(COMPLEX_022, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "complex_022.tab", "\t")
// END: COMPLEX_022

// BEGIN: COMPLEX_025
inserter = data.getInserter(COMPLEX_025, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "complex_025.tab", "\t")
// END: COMPLEX_025

// BEGIN: COMPLEX_026
inserter = data.getInserter(COMPLEX_026, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "complex_026.tab", "\t")
// END: COMPLEX_026

// BEGIN: COMPLEX_031
inserter = data.getInserter(COMPLEX_031, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "complex_031.tab", "\t")
// END: COMPLEX_031

// BEGIN: COMPLEX_033
inserter = data.getInserter(COMPLEX_033, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "complex_033.tab", "\t")
// END: COMPLEX_033

// BEGIN: COMPLEX_034
inserter = data.getInserter(COMPLEX_034, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "complex_034.tab", "\t")
// END: COMPLEX_034

// BEGIN: COMPLEX_035
inserter = data.getInserter(COMPLEX_035, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "complex_035.tab", "\t")
// END: COMPLEX_035

// BEGIN: COMPLEX_042
inserter = data.getInserter(COMPLEX_042, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "complex_042.tab", "\t")
// END: COMPLEX_042

// BEGIN: COMPLEX_044
inserter = data.getInserter(COMPLEX_044, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "complex_044.tab", "\t")
// END: COMPLEX_044

// BEGIN: COMPLEX_048
inserter = data.getInserter(COMPLEX_048, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "complex_048.tab", "\t")
// END: COMPLEX_048

// BEGIN: COMPLEX_052
inserter = data.getInserter(COMPLEX_052, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "complex_052.tab", "\t")
// END: COMPLEX_052

// BEGIN: COMPLEX_062
inserter = data.getInserter(COMPLEX_062, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "complex_062.tab", "\t")
// END: COMPLEX_062



// BEGIN: FAMILY_001
inserter = data.getInserter(FAMILY_001, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_001.tab", "\t")
// END: FAMILY_001

// BEGIN: FAMILY_002
inserter = data.getInserter(FAMILY_002, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_002.tab", "\t")
// END: FAMILY_002

// BEGIN: FAMILY_003
inserter = data.getInserter(FAMILY_003, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_003.tab", "\t")
// END: FAMILY_003

// BEGIN: FAMILY_004
inserter = data.getInserter(FAMILY_004, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_004.tab", "\t")
// END: FAMILY_004

// BEGIN: FAMILY_005
inserter = data.getInserter(FAMILY_005, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_005.tab", "\t")
// END: FAMILY_005

// BEGIN: FAMILY_006
inserter = data.getInserter(FAMILY_006, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_006.tab", "\t")
// END: FAMILY_006

// BEGIN: FAMILY_007
inserter = data.getInserter(FAMILY_007, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_007.tab", "\t")
// END: FAMILY_007

// BEGIN: FAMILY_008
inserter = data.getInserter(FAMILY_008, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_008.tab", "\t")
// END: FAMILY_008

// BEGIN: FAMILY_009
inserter = data.getInserter(FAMILY_009, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_009.tab", "\t")
// END: FAMILY_009

// BEGIN: FAMILY_010
inserter = data.getInserter(FAMILY_010, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_010.tab", "\t")
// END: FAMILY_010

// BEGIN: FAMILY_011
inserter = data.getInserter(FAMILY_011, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_011.tab", "\t")
// END: FAMILY_011

// BEGIN: FAMILY_012
inserter = data.getInserter(FAMILY_012, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_012.tab", "\t")
// END: FAMILY_012

// BEGIN: FAMILY_013
inserter = data.getInserter(FAMILY_013, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_013.tab", "\t")
// END: FAMILY_013

// BEGIN: FAMILY_014
inserter = data.getInserter(FAMILY_014, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_014.tab", "\t")
// END: FAMILY_014

// BEGIN: FAMILY_015
inserter = data.getInserter(FAMILY_015, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_015.tab", "\t")
// END: FAMILY_015

// BEGIN: FAMILY_016
inserter = data.getInserter(FAMILY_016, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_016.tab", "\t")
// END: FAMILY_016

// BEGIN: FAMILY_017
inserter = data.getInserter(FAMILY_017, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_017.tab", "\t")
// END: FAMILY_017

// BEGIN: FAMILY_018
inserter = data.getInserter(FAMILY_018, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_018.tab", "\t")
// END: FAMILY_018

// BEGIN: FAMILY_019
inserter = data.getInserter(FAMILY_019, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_019.tab", "\t")
// END: FAMILY_019

// BEGIN: FAMILY_020
inserter = data.getInserter(FAMILY_020, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_020.tab", "\t")
// END: FAMILY_020

// BEGIN: FAMILY_021
inserter = data.getInserter(FAMILY_021, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_021.tab", "\t")
// END: FAMILY_021

// BEGIN: FAMILY_024
inserter = data.getInserter(FAMILY_024, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_024.tab", "\t")
// END: FAMILY_024

// BEGIN: FAMILY_025
inserter = data.getInserter(FAMILY_025, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_025.tab", "\t")
// END: FAMILY_025

// BEGIN: FAMILY_029
inserter = data.getInserter(FAMILY_029, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_029.tab", "\t")
// END: FAMILY_029

// BEGIN: FAMILY_030
inserter = data.getInserter(FAMILY_030, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_030.tab", "\t")
// END: FAMILY_030

// BEGIN: FAMILY_031
inserter = data.getInserter(FAMILY_031, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_031.tab", "\t")
// END: FAMILY_031

// BEGIN: FAMILY_032
inserter = data.getInserter(FAMILY_032, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_032.tab", "\t")
// END: FAMILY_032

// BEGIN: FAMILY_033
inserter = data.getInserter(FAMILY_033, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_033.tab", "\t")
// END: FAMILY_033

// BEGIN: FAMILY_034
inserter = data.getInserter(FAMILY_034, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_034.tab", "\t")
// END: FAMILY_034

// BEGIN: FAMILY_037
inserter = data.getInserter(FAMILY_037, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_037.tab", "\t")
// END: FAMILY_037

// BEGIN: FAMILY_038
inserter = data.getInserter(FAMILY_038, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_038.tab", "\t")
// END: FAMILY_038

// BEGIN: FAMILY_045
inserter = data.getInserter(FAMILY_045, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_045.tab", "\t")
// END: FAMILY_045

// BEGIN: FAMILY_046
inserter = data.getInserter(FAMILY_046, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_046.tab", "\t")
// END: FAMILY_046

// BEGIN: FAMILY_052
inserter = data.getInserter(FAMILY_052, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_052.tab", "\t")
// END: FAMILY_052

// BEGIN: FAMILY_054
inserter = data.getInserter(FAMILY_054, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_054.tab", "\t")
// END: FAMILY_054

// BEGIN: FAMILY_057
inserter = data.getInserter(FAMILY_057, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_057.tab", "\t")
// END: FAMILY_057

// BEGIN: FAMILY_058
inserter = data.getInserter(FAMILY_058, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_058.tab", "\t")
// END: FAMILY_058

// BEGIN: FAMILY_063
inserter = data.getInserter(FAMILY_063, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_063.tab", "\t")
// END: FAMILY_063

// BEGIN: FAMILY_193
inserter = data.getInserter(FAMILY_193, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_193.tab", "\t")
// END: FAMILY_193

// BEGIN: FAMILY_069
inserter = data.getInserter(FAMILY_069, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_069.tab", "\t")
// END: FAMILY_069

// BEGIN: FAMILY_083
inserter = data.getInserter(FAMILY_083, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "family_083.tab", "\t")
// END: FAMILY_083

// BEGIN: FAMILY_374
// inserter = data.getInserter(FAMILY_374, dummy_tr)
// InserterUtils.loadDelimitedData(inserter, traindir + "family_374.tab", "\t")
// END: FAMILY_374


/*
 * Testing split for model inference
 * Observed partitions
 */

//def testdir = 'data'+java.io.File.separator+ foldStr + 'test'+java.io.File.separator;
def testdir = 'data' + java.io.File.separator + 'test' + java.io.File.separator;

inserter = data.getInserter(ExpUp, observed_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir + "ExpUp.csv", ",");

inserter = data.getInserter(ExpDown, observed_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir + "ExpDown.csv", ",");

inserter = data.getInserter(MutPlus, observed_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir + "MutPlus.csv", ",");

inserter = data.getInserter(MutMinus, observed_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir + "MutMinus.csv", ",");

/*
 * Random variable partitions
 */

inserter = data.getInserter(TargetPatientLabel, PatientLabelTruth)
InserterUtils.loadDelimitedDataTruth(inserter, testdir + "PatientLabel.csv", ",");

/* ?? predicates on latents ?? */

inserter = data.getInserter(Activates, dummy_te)
InserterUtils.loadDelimitedData(inserter, testdir + "Activates.csv", ",")

inserter = data.getInserter(Inhibits, dummy_te)
InserterUtils.loadDelimitedData(inserter, testdir + "Inhibits.csv", ",")

inserter = data.getInserter(Similar, dummy_te)
InserterUtils.loadDelimitedData(inserter, testdir + "Similar.csv", ",")

/*to populate testDB with the correct rvs
 */

inserter = data.getInserter(TargetPatientLabel, dummy_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir + "BogusPatientLabel.csv", ",");


/*
 * Set up training databases for weight learning using training set
 */

Database distributionDB = data.getDatabase(predict_tr, [ExpUp, ExpDown, MutPlus, MutMinus] as Set, observed_tr);
Database truthDB = data.getDatabase(truth_tr, [TargetPatientLabel] as Set)
Database dummy_DB = data.getDatabase(dummy_tr, [Activates, Inhibits, Similar, TargetPatientLabel] as Set)

/* Populate distribution DB. */
DatabasePopulator dbPop = new DatabasePopulator(distributionDB);
dbPop.populateFromDB(dummy_DB, Activates);
dbPop.populateFromDB(dummy_DB, Inhibits);
dbPop.populateFromDB(dummy_DB, Similar);
dbPop.populateFromDB(dummy_DB, TargetPatientLabel);


/*
HardEM weightLearning = new HardEM(model, distributionDB, truthDB, cb);
println "about to start weight learning"
weightLearning.learn();
println " finished weight learning "
weightLearning.close();
*/
/*
 MaxPseudoLikelihood mple = new MaxPseudoLikelihood(model, trainDB, truthDB, cb);
 println "about to start weight learning"
 mple.learn();
 println " finished weight learning "
 mlpe.close();
 */

println model;


distributionDB.close()
truthDB.close()
dummy_DB.close()
