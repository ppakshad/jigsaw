import soot.*;
import soot.options.Options;
import soot.toolkits.graph.Block;
import soot.toolkits.graph.ExceptionalUnitGraph;
import soot.toolkits.graph.pdg.HashMutablePDG;
import soot.toolkits.graph.pdg.PDGNode;
import soot.util.Chain;

import java.io.*;
import java.util.*;

public class Soot_utlilty {
    protected static int APK_API = 23;
    protected static String jarsPath = Instrumenter.AndroidSdk+"platforms/android-"+APK_API+"/android.jar";
    protected static List<String> excludePackagesList = new ArrayList<String>();
    protected static List<String> excludeLibrariesList = new ArrayList<String>();
    protected static List<String> excludeMethodList = new ArrayList<String>();



    static
    {
        excludePackagesList.add("java.");
        excludePackagesList.add("android.");
        excludePackagesList.add("javax.");
        excludePackagesList.add("sun.");
        excludePackagesList.add("android.support.");
        excludePackagesList.add("junit.");
        excludePackagesList.add("org.w3c");
        excludePackagesList.add("org.xmlpull");
        excludePackagesList.add("org.xml.sax.");
        excludePackagesList.add("org.json");
        excludePackagesList.add("org.apache.http.");
        excludePackagesList.add("com.google.android");
        excludePackagesList.add("com.android.");
        excludePackagesList.add("int");
        excludePackagesList.add("String");
        excludePackagesList.add("dalvik.");
        excludePackagesList.add("byte");
        excludePackagesList.add("boolean");
        excludePackagesList.add("short");
        excludePackagesList.add("long");
        excludePackagesList.add("char");
        excludePackagesList.add("void");
        excludePackagesList.add("double");
        excludePackagesList.add("float");
        excludePackagesList.add("null");
        excludePackagesList.add("Slice");

    }

    static{
        excludeLibrariesList.add("java.");
        excludeLibrariesList.add("android.");
        excludeLibrariesList.add("sun.");
        excludeLibrariesList.add("javax.");
        excludeLibrariesList.add("android.support.");
        excludeLibrariesList.add("junit.");
        excludeLibrariesList.add("org.w3c");
        excludeLibrariesList.add("org.xmlpull");
        excludeLibrariesList.add("org.xml.sax.");
        excludeLibrariesList.add("org.json");
        excludeLibrariesList.add("org.apache.http.");
        excludeLibrariesList.add("com.google.android");
        excludeLibrariesList.add("com.android.");
    }

    static{
        excludeMethodList.add("<clinit>");
        excludeMethodList.add("<init>");
    }


    boolean isExcludeClass(SootClass sootClass)
    {
        if (sootClass.isPhantom())
        {
            return true;
        }

        String packageName = sootClass.getPackageName();
        for (String exclude : excludePackagesList)
        {
            if (packageName.startsWith(exclude)  && !packageName.startsWith("android.support."))
            {
                return true;
            }
        }

        return false;
    }

    boolean isExcludeClass(String sootClass)
    {
        if(Scene.v().getSootClass(sootClass).isPhantom()){
            return true;
        }

        for (String exclude : excludePackagesList)
        {
            if (sootClass.startsWith(exclude)  && !sootClass.startsWith("android.support.") )
            {
                return true;
            }
        }

        return false;
    }

    boolean isExcludedLibrary(String Library)
    {
        if(Scene.v().getSootClass(Library).isPhantom()){
            return true;
        }


        for (String exclude : excludeLibrariesList)
        {
            if (Library.startsWith(exclude)  && !Library.startsWith("android.support.") )
            {
                return true;
            }
        }

        return false;
    }

    boolean isExcludedLibrary_noASexc(String Library)
    {
        if(Scene.v().getSootClass(Library).isPhantom()){
            return true;
        }


        for (String exclude : excludeLibrariesList)
        {
            if (Library.startsWith(exclude) )
            {
                return true;
            }
        }

        return false;
    }

    public boolean get_if_ancient (SootClass c,String comparison){
        SootClass tmp= c;
        while(tmp.hasSuperclass()){
            if(!tmp.getSuperclass().getName().equals(comparison)) {
                tmp = tmp.getSuperclass();
            }else{
                return true;
            }
        }
        return false;
    }




    public File[] get_directories(String path){
        File dir = new File(path);
        File[] files = dir.listFiles();
        FileFilter fileFilter = new FileFilter() {
            public boolean accept(File file) {
                return file.isDirectory();
            }
        };
        files = dir.listFiles(fileFilter);
        return files;
    }




    public void initSoot(String apkPath, File dexPath){

        Options.v().set_src_prec(Options.src_prec_apk_c_j);

        Options.v().set_output_format(Options.output_format_dex);

        ArrayList<String> paths = new ArrayList<>();


        paths.add(apkPath);
        paths.add(jarsPath);


        paths.add(dexPath.getAbsolutePath());
        Options.v().force_overwrite();

        Options.v().set_process_dir(paths);

        Options.v().set_force_android_jar(apkPath);

        Options.v().set_keep_line_number(true);

        Options.v().set_process_multiple_dex(true);

        Options.v().set_allow_phantom_refs(true);

        Options.v().set_whole_program(true);

        Options.v().set_android_api_version(APK_API);

        Options.v().set_wrong_staticness(Options.wrong_staticness_fix);


        Scene.v().loadNecessaryClasses();

        PackManager.v().runPacks();

    }


    public void exclude_slices(){
        for(Iterator<SootClass> iter = Scene.v().getApplicationClasses().snapshotIterator(); iter.hasNext();){
            SootClass c = iter.next();
            if(c.getName().startsWith("Slice")){
                c.setLibraryClass();
            }
            if(this.isExcludedLibrary(c.getName())){
                c.setLibraryClass();
            }

        }
    }


    boolean isExcludeClassNOASEX(SootClass sootClass)
    {
        if (sootClass.isPhantom())
        {
            return true;
        }

        String packageName = sootClass.getPackageName();
        for (String exclude : excludePackagesList)
        {
            if (packageName.startsWith(exclude))
            {
                return true;
            }
        }

        return false;
    }


    public Map<Integer, ArrayList<SootClass>> get_scores(ArrayList<SootClass> malwares ){
        Map<Integer, ArrayList<SootClass>> to_ret = new HashMap<>();
        Jimple_utility ju = new Jimple_utility();
        for(SootClass sc : malwares){
            if(!isExcludeClassNOASEX(sc)) {
                int CC = ju.calculate_cyclomatic_complexity(sc);

                if (to_ret.containsKey(CC)) {
                    to_ret.get(CC).add(sc);
                } else {
                    ArrayList<SootClass> tmp = new ArrayList<>();
                    tmp.add(sc);
                    to_ret.put(CC, tmp);
                }

            }
        }
        return  to_ret;
    }

    public double get_AVG(Map<Integer, ArrayList<SootClass>> map, int size){
        int result =0;
        int real_size = size;
        for(int i : map.keySet()){
            ArrayList<SootClass> tmp_values = map.get(i);
            if(i==0){
                real_size = real_size-tmp_values.size();
            }else {

                int tmp_result = i * tmp_values.size();
                result += tmp_result;
            }
        }
        return result/real_size;

    }

    public void  add_dependencies(My_slice slice){
        ArrayList<Unit> units = slice.getMy_units();
        ArrayList<Local> locals = slice.getMy_locals();
        for(Unit u : units){
            for(ValueBox v : u.getUseAndDefBoxes()){
                if(!isExcludeClass(v.getValue().getType().toString())) {
                    for(String s : extract_activity_dependencies(new ArrayList<String>(),v.getValue().getType().toString())){
                        SootClass sc = Scene.v().getSootClass(s);
                        sc.setApplicationClass();
                    }
                }
            }
        }
        for(Local l : locals){
            if(!isExcludeClass(l.getType().toString())) {
                for(String s : extract_activity_dependencies(new ArrayList<String>(),l.getType().toString())){
                    SootClass sc = Scene.v().getSootClass(s);
                    sc.setApplicationClass();
                }
            }
        }
    }

    public ArrayList<String> extract_activity_dependencies(ArrayList<String> dependencies, String feature) {
        ArrayList<String> new_dep = new ArrayList<String>();
        Soot_utlilty utility = new Soot_utlilty();
        if(!dependencies.contains(feature)){
            dependencies.add(feature);
        }
        for (SootClass c : Scene.v().getApplicationClasses()) {
            if(c.getName().equals(feature)){
                Chain<SootClass> interfaces = c.getInterfaces();
                Chain<SootField> fields = c.getFields();
                List<SootMethod> methods = c.getMethods();

                if(c.hasSuperclass()){
                    SootClass superclass = c.getSuperclass();
                    if (!utility.isExcludeClass(superclass)) {
                        new_dep.add(superclass.getName());
                    }
                }


                for(Iterator<SootField> iter = fields.iterator(); iter.hasNext();){
                    SootField field= iter.next();
                    if (!utility.isExcludeClass(field.getType().toString()) && !field.getType().toString().contains("[]")) {
                        if (!dependencies.contains(field.getType().toString()) && !new_dep.contains(field.getType().toString())) {
                            new_dep.add(field.getType().toString());
                        }
                    }

                }
                for(Iterator<SootClass> iter = interfaces.iterator(); iter.hasNext();){
                    SootClass field= iter.next();
                    if (!utility.isExcludeClass(field.getType().toString())) {
                        if (!dependencies.contains(field.getType().toString()) && !new_dep.contains(field.getType().toString())) {
                            new_dep.add(field.getType().toString());
                        }
                    }

                }

                for (SootMethod m : methods ){
                    if (!m.hasActiveBody()) {
                        continue;
                    }
                    Body b = m.getActiveBody();
                    boolean able =true;
                    try{
                        ExceptionalUnitGraph CFG = new ExceptionalUnitGraph(b);
                        HashMutablePDG PDG = new HashMutablePDG(CFG);
                        for(Iterator <PDGNode> iter = PDG.iterator(); iter.hasNext();){
                            PDGNode node = iter.next();
                            for (PDGNode a : node.getDependents()){
                                Block block = (Block) a.getNode();
                                for (Iterator<Unit> iter_u = block.iterator(); iter_u.hasNext();){
                                    Unit unit = iter_u.next();
                                    for (ValueBox v : unit.getUseAndDefBoxes()){
                                        String tmp_feat =v.getValue().getType().toString();
                                        if(!dependencies.contains(tmp_feat) && !utility.isExcludeClass(tmp_feat) && !new_dep.contains(tmp_feat)  && !tmp_feat.contains("[")){
                                            new_dep.add(tmp_feat);
                                        }
                                        String tmp =v.getValue().toString();
                                        if (tmp.startsWith("class")){
                                            String dep = tmp.split("\"")[1].split(";")[0].substring(1).replace("/",".");
                                            if (!utility.isExcludeClass(dep) && !dep.contains("[]") && dep!= feature){
                                                if(!dep.equals(c.getName()) ){
                                                    if(!dependencies.contains(dep) && !new_dep.contains(dep) ){
                                                        new_dep.add(dep);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                            }
                        }
                    }catch(Exception e){
                        able = false;
                    }
                    if (able == false){
                        for (Iterator<Local> iter_local = b.getLocals().iterator(); iter_local.hasNext();){
                            Local local_tmp = iter_local.next();
                            if (!utility.isExcludeClass(local_tmp.getType().toString()) && !local_tmp.getType().toString().contains("[]")){
                                if(!local_tmp.getType().toString().equals(c.getName()) ){
                                    if(!dependencies.contains(local_tmp.getType().toString()) && !new_dep.contains(local_tmp.getType().toString())){
                                        new_dep.add(local_tmp.getType().toString());
                                    }
                                }
                            }
                        }
                        for (Iterator<ValueBox> iter_units = b.getUseAndDefBoxes().iterator() ; iter_units.hasNext();){
                            ValueBox value = iter_units.next();
                            String tmp_feat =value.getValue().getType().toString();
                            if(!dependencies.contains(tmp_feat) && !utility.isExcludeClass(tmp_feat) && !new_dep.contains(tmp_feat)  && !tmp_feat.contains("[")){
                                new_dep.add(tmp_feat);
                            }
                            String tmp =value.getValue().toString();
                            if (tmp.startsWith("class")){
                                String dep = tmp.split("\"")[1].split(";")[0].substring(1).replace("/",".");
                                if (!utility.isExcludeClass(dep) && !dep.contains("[]") && dep!= feature){
                                    if(!dep.equals(c.getName()) ){
                                        if(!dependencies.contains(dep) && !new_dep.contains(dep) ){
                                            new_dep.add(dep);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if (new_dep.size() >0){
            dependencies.addAll(new_dep);
            for (String i : new_dep){
                if (i != feature){
                    ArrayList<String> tmp_dep = extract_activity_dependencies(dependencies,i);
                    if(tmp_dep.size() >0) {
                        for (String s : tmp_dep) {
                            if (!dependencies.contains(s)) {
                                dependencies.add(s);
                            }
                        }
                    }
                }
            }
        }
        if(!dependencies.contains(feature)) {
            dependencies.add(feature);
        }
        Set<String> foo = new HashSet<String>(dependencies);
        ArrayList<String> mainList = new ArrayList<String>();
        mainList.addAll(foo);
        return mainList;
    }



    public ArrayList<SootClass> find_malware_classes(Map<String, ArrayList<String>> goodwares_classes) {
        ArrayList<String> tmp_all_classes = new ArrayList<>();
        ArrayList<SootClass> to_return = new ArrayList<>();
        for(String s : goodwares_classes.keySet()){
            tmp_all_classes.addAll(goodwares_classes.get(s));
        }
        for(SootClass c : Scene.v().getClasses()){
            if(this.isExcludedLibrary(c.getName()) || c.isPhantom() || c.isPhantomClass() ){
                continue;
            }else{
                if(!tmp_all_classes.contains(c.getName())){
                    to_return.add(c);
                }
            }
        }
        return to_return;
    }

    public boolean isExcludedMethod(String name) {
        for (String exclude : excludeMethodList)
        {
            if (name.startsWith(exclude) )
            {
                return true;
            }
        }
        return false;
    }

    public ArrayList<String> get_goodware_classes(File i) {
        String fulldir = i.getAbsolutePath();
        ArrayList<String> to_ret = new ArrayList<>();
        File list_classes = new File(fulldir + "/classes.txt");
        if (list_classes.exists()) {
            try (BufferedReader br = new BufferedReader(new FileReader(list_classes))) {
                for (String line; (line = br.readLine()) != null; ) {
                    to_ret.add(line);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        to_ret.add("Slice"+i.getName());
        to_ret.add("Slice"+i.getName()+"method");
        return to_ret;

    }




}
