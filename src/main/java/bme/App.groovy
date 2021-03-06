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
import edu.umd.cs.psl.model.predicate.StandardPredicate
import edu.umd.cs.psl.ui.loading.*
import edu.umd.cs.psl.util.database.Queries
import edu.ucsc.cs.utils.Evaluator;
import edu.ucsc.cs.utils.Dumper;

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
// TargetPatient(P) -- OBSERVED (target var. flag)
model.add predicate: "TargetPatient" , types:[ArgumentType.UniqueID]
// TargetPatientLabel(P) -- target var.
model.add predicate: "TargetPatientLabel" , types:[ArgumentType.UniqueID]

/*
 * Evidence (observed) predicates
 */

// ExpUp(G, P) - observed
model.add predicate: "ExpUp" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID]
// ExpDown(G, P) - observed
model.add predicate: "ExpDown" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID]
// model.add predicate: "MutPlus" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID]  // GOF
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
// model.add rule : MutPlus(G,P) >> ~ActiveObs(G, P), weight : 1
model.add rule : MutMinus(G, P) >> ~Active(G, P), weight : 1

// rules to relate gene activities via gene interaction network
model.add rule : (Inhibits(G1, G2) & Active(G1, P)) >> ~Active(G2, P), weight : 1
model.add rule : (Activates(G1, G2) & Active(G1, P)) >> Active(G2, P), weight : 1
model.add rule : (Similar(G1, G2) & GeneLabel(G1)) >> GeneLabel(G2), weight : 1

// label inference
model.add rule : (PatientLabel(P) & Active(G, P)) >> GeneLabel(G), weight : 1
model.add rule : (TargetPatient(P) & GeneLabel(G) & Active(G, P)) >> TargetPatientLabel(P), weight : 1
// prior:
// model.add rule: ~TargetPatientLabel(P), weight : 1


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
def traindir = 'data/Gleason_score' + java.io.File.separator + 'Train' + java.io.File.separator;

/*
 * Observed data partition
 * 
 */

inserter = data.getInserter(ExpUp, observed_tr)
InserterUtils.loadDelimitedDataTruth(inserter, traindir + "ExpUp.tab", "\t");

inserter = data.getInserter(ExpDown, observed_tr)
InserterUtils.loadDelimitedDataTruth(inserter, traindir + "ExpDown.tab", "\t");

// inserter = data.getInserter(MutPlus, observed_tr)
// InserterUtils.loadDelimitedDataTruth(inserter, traindir + "MutPlus.csv", ",");

inserter = data.getInserter(MutMinus, observed_tr)
InserterUtils.loadDelimitedDataTruth(inserter, traindir + "MutMinus.tab", "\t");

inserter = data.getInserter(TargetPatient, observed_tr)
InserterUtils.loadDelimitedDataTruth(inserter, traindir + "TargetPatient.tab", "\t")

inserter = data.getInserter(PatientLabel, observed_tr)
InserterUtils.loadDelimitedDataTruth(inserter, traindir + "PatientLabel.tab", "\t");

inserter = data.getInserter(Activates, observed_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "Activates.tab", "\t")
// COMPLEX
InserterUtils.loadDelimitedDataTruth(inserter, traindir + "Complex.tab", "\t")
// FAMILY
InserterUtils.loadDelimitedDataTruth(inserter, traindir + "Family.tab", "\t")

inserter = data.getInserter(Inhibits, observed_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "Inhibits.tab", "\t")

// inserter = data.getInserter(Similar, observed_tr)
// InserterUtils.loadDelimitedData(inserter, traindir + "Similar.csv", ",")


/*
 * Ground truth for training data for weight learning
 */

inserter = data.getInserter(TargetPatientLabel, truth_tr)
InserterUtils.loadDelimitedDataTruth(inserter, traindir + "TargetPatientLabel.tab", "\t")


/*
 * Used later on to populate training DB with all possible interactions
 * ??? predicates on latents ???
 */

inserter = data.getInserter(Active, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "Active.tab", "\t")

inserter = data.getInserter(GeneLabel, dummy_tr)
InserterUtils.loadDelimitedData(inserter, traindir + "GeneLabel.tab", "\t")

inserter = data.getInserter(TargetPatientLabel, dummy_tr)
InserterUtils.loadDelimitedDataTruth(inserter, traindir + "BogusPatientLabel.tab", "\t")


/*
 * Set up training databases for weight learning using training set
 */

Set closedPredicates_tr = [ExpUp, ExpDown, MutPlus, MutMinus, TargetPatient, PatientLabel, Activates, Inhibits, Similar] as Set;
Database distributionDB = data.getDatabase(predict_tr, closedPredicates_tr, observed_tr);
Database truthDB = data.getDatabase(truth_tr, [TargetPatientLabel] as Set)
Database dummy_DB = data.getDatabase(dummy_tr, [Active, GeneLabel, TargetPatientLabel] as Set)

/* Populate distribution DB. */
DatabasePopulator dbPop = new DatabasePopulator(distributionDB);
dbPop.populateFromDB(dummy_DB, TargetPatient);
dbPop.populateFromDB(dummy_DB, Active);
dbPop.populateFromDB(dummy_DB, GeneLabel);
dbPop.populateFromDB(dummy_DB, TargetPatientLabel);


HardEM weightLearning = new HardEM(model, distributionDB, truthDB, cb);
println "about to start weight learning"
weightLearning.learn();
println " finished weight learning "
weightLearning.close();
/*
 MaxPseudoLikelihood mple = new MaxPseudoLikelihood(model, trainDB, truthDB, cb);
 println "about to start weight learning"
 mple.learn();
 println " finished weight learning "
 mlpe.close();
 */

//
Dumper dumper = new Dumper(distributionDB, GeneLabel, "GeneLabel_distributionDB.csv");
dumper.outputToFile();
//
dumper = new Dumper(dummy_DB, GeneLabel, "GeneLabel_dummy_DB.csv");
dumper.outputToFile();

println model;


/////////////////////////////////////////////////////////////////////////


/*
 * Testing split for model inference
 * Observed partitions
 */

//def testdir = 'data'+java.io.File.separator+ foldStr + 'test'+java.io.File.separator;
def testdir = 'data/Gleason_score' + java.io.File.separator + 'Test' + java.io.File.separator;

inserter = data.getInserter(TargetPatient, observed_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir + "TargetPatient.tab", "\t")

inserter = data.getInserter(ExpUp, observed_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir + "ExpUp.tab", "\t");

inserter = data.getInserter(ExpDown, observed_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir + "ExpDown.tab", "\t");

inserter = data.getInserter(MutMinus, observed_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir + "MutMinus.tab", "\t");

// PATHWAY
inserter = data.getInserter(Activates, observed_te)
InserterUtils.loadDelimitedData(inserter, traindir + "Activates.tab", "\t")
// COMPLEX
InserterUtils.loadDelimitedDataTruth(inserter, traindir + "Complex.tab", "\t")
// FAMILY
InserterUtils.loadDelimitedDataTruth(inserter, traindir + "Family.tab", "\t")
//
inserter = data.getInserter(Inhibits, observed_te)
InserterUtils.loadDelimitedData(inserter, traindir + "Inhibits.tab", "\t")


//
// load from Dump!
//
inserter = data.getInserter(GeneLabel, observed_te)
InserterUtils.loadDelimitedDataTruth(inserter, "./" + "GeneLabel_distributionDB.csv", ",");

/*
 * Random variable partitions
 */

inserter = data.getInserter(TargetPatientLabel, PatientLabelTruth)
InserterUtils.loadDelimitedDataTruth(inserter, testdir + "PatientLabel.tab", "\t");

/* ?? predicates on latents ?? */

inserter = data.getInserter(Active, dummy_te)
InserterUtils.loadDelimitedData(inserter, testdir + "Active.tab", "\t")

inserter = data.getInserter(TargetPatientLabel, dummy_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir + "BogusPatientLabel.tab", "\t")

/*to populate testDB with the correct rvs
 */

Set closedPredicates_te = [ExpUp, ExpDown, MutPlus, MutMinus, TargetPatient, PatientLabel, Activates, Inhibits, Similar, GeneLabel] as Set;
Database testDB = data.getDatabase(predict_te, closedPredicates_te, observed_te);
Database testTruth_PatientLabel = data.getDatabase(PatientLabelTruth, [TargetPatientLabel] as Set)
Database dummy_test = data.getDatabase(dummy_te, [Active, TargetPatientLabel] as Set)

/* Populate in test DB. */

DatabasePopulator test_populator = new DatabasePopulator(testDB);
test_populator.populateFromDB(dummy_test, Active);
test_populator.populateFromDB(dummy_test, TargetPatientLabel);
//
// from train??!!!
//
test_populator.populateFromDB(distributionDB, GeneLabel);
//
dumper = new Dumper(testDB, GeneLabel, "GeneLabel_testDB.csv");
dumper.outputToFile();
// test_populator.populateFromDB(dummy_DB, Activates);
// test_populator.populateFromDB(dummy_DB, Inhibits);


/*
 * Inference
 */

MPEInference mpe = new MPEInference(model, testDB, cb)
FullInferenceResult result = mpe.mpeInference()
System.out.println("Objective: " + result.getTotalWeightedIncompatibility())


Evaluator evaluator = new Evaluator(testDB, testTruth_PatientLabel, TargetPatientLabel);
evaluator.outputToFile();


//
distributionDB.close()
truthDB.close()
dummy_DB.close()
//
testDB.close()
testTruth_PatientLabel.close()
dummy_test.close()
