import org.xmlpull.v1.XmlPullParserException;
import soot.*;
import soot.jimple.infoflow.android.manifest.IComponentContainer;
import soot.jimple.infoflow.android.manifest.ProcessManifest;
import soot.jimple.infoflow.android.manifest.binary.BinaryManifestBroadcastReceiver;
import soot.toolkits.graph.Block;
import soot.toolkits.graph.ExceptionalBlockGraph;
import soot.toolkits.graph.ExceptionalUnitGraph;

import java.io.IOException;
import java.util.*;

public class ReceiverExtractor implements Extractor {

    private Soot_utlilty utility = new Soot_utlilty();
    private String SEARCHED = "sendBroadcast";

    public void extractReceiver(String feature) {
        try {
            ProcessManifest manifest = new ProcessManifest(Instrumenter.apkPath);
            IComponentContainer<BinaryManifestBroadcastReceiver> receivers = manifest.getBroadcastReceivers();
            for(BinaryManifestBroadcastReceiver node : receivers){
                if(node.toString().contains(feature)){
                    String tmp_str = node.toString();
                    String type = tmp_str.split("<")[1].split(" ")[0];
                    String new_tmp = tmp_str.concat("</"+type+">");
                    this.utility.WriteFile("xml_tag.xml",new_tmp);
                    break;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        } catch (XmlPullParserException e) {
            e.printStackTrace();
        }
    }

    @Override
    public SootClass extract_feature(String feature, String name_folder) {
        SootClass slice_class = null;
        ArrayList<String> dependencies = new ArrayList<>();
        if(!feature.startsWith(".")) {
            dependencies = utility.extract_activity_dependencies_PDG(new ArrayList<String>(), feature);
            System.out.println(dependencies.size());

        }else{
            if(Instrumenter.DEBUG) {
                System.out.println("DEBUG : Relative declaration of receiver.\nTrying with soft dependencies check ..");
            }
            dependencies = utility.extract_activity_dependencies_PDG_soft(new ArrayList<String>(), feature);
            if(!dependencies.isEmpty()){
                String correspondence ="";
                for(String s : dependencies){
                    if(s.contains(feature) && !s.contains("$") && !s.equals(feature)){
                        correspondence= s +":"+feature;
                        if(Instrumenter.DEBUG) {
                            System.out.println("DEBUG : Real name "+s);
                        }
                        feature= s;
                        break;
                    }
                }
                utility.WriteFile("Relative_features.txt",correspondence);

            }else{
                System.out.println("No feature found :(((( ");
            }
        }
        ArrayList<My_slice> slices = this.identify_sendBroadcast(feature);
        System.out.println(slices.size());
        if (!slices.isEmpty()) {
            My_slice simplest = null;
            if (slices.size() > 1) {
                simplest = utility.get_simpler_slice(slices);
            } else {
                simplest = slices.get(0);
            }
            simplest.setName("Slice" + name_folder + "receiver");
            simplest.setFeature(feature);
            ArrayList<String> slice_dep = utility.create_dependencies_file(simplest);
            dependencies.addAll(slice_dep);
            SootClass tmp = simplest.get_soot_class();
            System.out.println(simplest.toString());
            utility.WriteFile("class_of_extraction.txt", simplest.getFeature());
            slice_class = tmp;
        } else {
            System.out.println("No possible slice found :(");
        }
        utility.extract_classes(dependencies);

        return slice_class;
    }




    public ArrayList<My_slice> identify_sendBroadcast(String feature) {

        ArrayList<String> soot_classes = new ArrayList<>();

        ArrayList<My_slice> soot_slice = new ArrayList<>();
        System.out.println("Trying to extract slice Service invocation : " + feature);
        for (SootClass c : Scene.v().getApplicationClasses()) {
            if (utility.isExcludeClass(c)) {
                continue;
            }
            for (SootMethod m : c.getMethods()) {
                if(!m.hasActiveBody()){
                    continue;
                }
                if (m.getActiveBody().toString().contains(SEARCHED)) {
                    Body b = m.getActiveBody();
                    if(Instrumenter.DEBUG) {
                        System.out.println("DEBUG : Getting CFG");
                    }
                    ExceptionalUnitGraph CFG = new ExceptionalUnitGraph(b);
                    ExceptionalBlockGraph BFG = new ExceptionalBlockGraph(CFG);
                    if(Instrumenter.DEBUG) {
                        System.out.println("DEBUG : Start searching  invocation into the CFG");
                    }
                    for(Iterator<Block> iter = BFG.iterator(); iter.hasNext();){
                        Block block = iter.next();
                        if(block.toString().contains(SEARCHED)){
                            if(Instrumenter.DEBUG) {
                                System.out.println("DEBUG : same block :D ");
                            }
                            ArrayList<Unit> units = new ArrayList<>();
                            for(Iterator<Unit> iter_block = block.iterator(); iter_block.hasNext();){
                                units.add(iter_block.next());
                            }
                            boolean completed = false;
                            ArrayList<Local> locals_to_export = new ArrayList<>();
                            while (!completed) {
                                ArrayList<String> locals_to_add = new ArrayList<>();
                                for (Iterator<Unit> iter_un = units.iterator(); iter_un.hasNext(); ) {
                                    Unit un = iter_un.next();
                                    List<ValueBox> def = un.getDefBoxes();
                                    for (Iterator<ValueBox> iter_def = def.iterator(); iter_def.hasNext(); ) {
                                        ValueBox def_ = iter_def.next();
                                        if (!locals_to_add.contains(def_.getValue().toString())) {
                                            locals_to_add.add(def_.getValue().toString());
                                        }
                                    }
                                }
                                for (Iterator<Local> iter_locals = m.getActiveBody().getLocals().iterator(); iter_locals.hasNext(); ) {
                                    Local tmp_local = iter_locals.next();
                                    if (locals_to_add.contains(tmp_local.getName()) && !locals_to_export.contains(tmp_local)) {
                                        locals_to_export.add(tmp_local);
                                    }
                                }
                                ArrayList<String> missing_names = utility.missing_values_units(units, locals_to_export);
                                if (!missing_names.isEmpty()) {
                                    ArrayList<Unit> missing_units = new ArrayList<>();
                                    for (String missing : missing_names) {
                                        for (Iterator<Unit> iter_un = block.getBody().getUnits().iterator(); iter_un.hasNext(); ) {
                                            Unit un = iter_un.next();
                                            List<ValueBox> def = un.getDefBoxes();
                                            if (!def.isEmpty()) {
                                                for (Iterator<ValueBox> iter_def = def.iterator(); iter_def.hasNext(); ) {
                                                    ValueBox def_ = iter_def.next();
                                                    if (def_.getValue().toString().equals(missing)) {
                                                        missing_units.add(un);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    units.addAll(0, missing_units);
                                }else{
                                    completed = true;
                                }

                            }
                            My_slice slice = new My_slice(locals_to_export,units);
                            slice.setFeature(c.getName());

                            soot_slice.add(slice);
                            if(!soot_classes.contains(c.getName())){
                                soot_classes.add(c.getName()+":"+m.getName());
                            }
                        }

                    }

                }
            }
        }
        for(My_slice sl : soot_slice){
            utility.add_dependencies(sl);
        }
        String to_out = "Feature : "+feature+"\n";
        for (String s : soot_classes){
            to_out= to_out +s +"\n";
        }
        if(Instrumenter.DEBUG) {
            System.out.println("DEBUG : writing file  ./slices_classes.txt ");
        }
        utility.WriteFile("slices_classes.txt", to_out);
        return soot_slice;
    }

    private ArrayList<Block> find_path_for_sendBroadcast(ArrayList<Block> visited,ArrayList<Block> final_path, Block b){

        if(b.toString().contains(SEARCHED)){
            if(!final_path.contains(b)) {
                final_path.add(b);
            }
        }else{
            visited.add(b);
            List<Block> succ = b.getSuccs();
            if(succ.size()>0) {
                for (Block block : succ) {
                    if(visited.contains(block)){
                        continue;
                    }else{
                        if(!final_path.contains(b)) {
                            final_path.add(b);
                        }
                        find_path_for_sendBroadcast(visited,final_path,block);
                    }
                }
            }
        }
        return  final_path;
    }

}
