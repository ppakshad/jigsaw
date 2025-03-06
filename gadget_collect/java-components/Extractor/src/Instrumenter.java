
import org.apache.commons.io.FileUtils;


import soot.*;

import soot.options.Options;

import java.io.*;
import java.util.*;

import static java.lang.System.exit;



public class Instrumenter {


    protected static String apkPath = "";

    private static String feature = "";

    private static String Feat_Type = "";

    public static String output_dir = "";


    protected static String jarsPath = "";


    public static int output_format = Options.output_format_jimple;

    public static boolean DEBUG = false;

    private static boolean slice_found = false;




    public static void main(String[] args) {

        if(args.length == 5){
            feature = args[0];
            apkPath = args[1];
            Feat_Type = args[2];
            output_dir = args[3];
            jarsPath = args[4]+"platforms";
        }else if(args.length == 6){
            feature = args[0];
            apkPath = args[1];
            Feat_Type = args[2];
            output_dir = args[3];
            jarsPath = args [4]+"platforms";
            DEBUG = Boolean.parseBoolean(args[5]);

        }else{
            System.out.println("Wrong arguments, invocation should be like:\njava -jar extractor.jar <feature> <path_to_goodware> <feature_type> <path_for_save_jimples>\n");
            exit(0);
        }

        Soot_utlilty config = new Soot_utlilty();
        String name_root_folder = "";
        if(Feat_Type.equals("URL")){
            // . become _
            // / become £
            // : become ;
             name_root_folder = feature.replace(".","_").replace("/","£").replace(":","^");
        }else if (Feat_Type.equals("intent") ||Feat_Type.equals("Activity") || Feat_Type.equals("service") || Feat_Type.equals("providers") || Feat_Type.equals("receiver") || Feat_Type.equals("permission")){
            name_root_folder = feature.replace(".","_");
        }else if(Feat_Type.equals("API_call")){
            name_root_folder = feature.replace("/","£");
        }else if (Feat_Type.equals("interesting_call")){
            name_root_folder=feature;
        }else{
            System.out.println("Feature "+Feat_Type+" not supported yet :(");
            exit(1);
        }
        name_root_folder = output_dir + name_root_folder;
        File folder = new File(name_root_folder);
        if(!folder.exists()){
            folder.mkdirs();
        }
        String[] get_name = apkPath.split("/");
        String name_folder = get_name[get_name.length - 1].split("\\.")[0];
        output_dir = name_root_folder + "/" + name_folder;
        config.initSoot(apkPath,output_dir);
        if(DEBUG){
            System.out.println("Extracting the feature "+feature+" from "+apkPath);
        }

        SootClass slice_class = null;
        if (Feat_Type.equals("Activity")) {
            System.out.println("The searched feature is an Activity");
            Activity_extractor activity_extractor = new Activity_extractor();
            slice_class = activity_extractor.extract_feature(feature, name_folder);
            if (slice_class != null) {
                slice_found = true;
            }

        }else if (Feat_Type.equals("URL")){
            System.out.println("The searched feature is a URL ");
            URL_extractor url_extractor = new URL_extractor();
            slice_class = url_extractor.extract_feature(feature, name_folder);
            if (slice_class != null) {
                slice_found = true;
            } else {
                System.out.println("Sorry, it was not possible to retrieve the intended feature from this application :( ");
                File dir = new File(output_dir);
                if (dir.isDirectory() && dir.exists()) {
                    try {
                        FileUtils.deleteDirectory(dir);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
                exit(0);
            }
        }else if (Feat_Type.equals("API_call")) {
            System.out.println("The searched feature is a api call ");
            ApiCallExtractor api_extractor = new ApiCallExtractor();
            slice_class = api_extractor.extract_feature(feature, name_folder);
            if (slice_class != null) {
                slice_found = true;
            } else {
                System.out.println("Sorry, it was not possible to retrieve the intended feature from this application :( ");
                File dir = new File(output_dir);
                if (dir.isDirectory() && dir.exists()) {
                    try {
                        FileUtils.deleteDirectory(dir);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
                exit(0);
            }

        }else if(Feat_Type.equals("interesting_call")){
            System.out.println("The searched feature is an interesting call ");
            CallExtractor call_extractor = new CallExtractor();
            slice_class = call_extractor.extract_feature(feature, name_folder);
            if (slice_class != null) {
                slice_found = true;
            } else {
                System.out.println("Sorry, it was not possible to retrieve the intended feature from this application :( ");
                File dir = new File(output_dir);
                if (dir.isDirectory() && dir.exists()) {
                    try {
                        FileUtils.deleteDirectory(dir);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
                exit(0);
            }
        }else if(Feat_Type.equals("service")){
            System.out.println("The searched feature is an service ");
            ServiceExtractor service_extractor = new ServiceExtractor();
            slice_class = service_extractor.extract_feature(feature, name_folder);
            if (slice_class != null) {
                slice_found = true;
                service_extractor.extractService(feature);
            } else {
                System.out.println("Sorry, it was not possible to retrieve the intended feature from this application :( ");
                File dir = new File(output_dir);
                if (dir.isDirectory() && dir.exists()) {
                    try {
                        FileUtils.deleteDirectory(dir);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
                exit(0);
            }
        }else if(Feat_Type.equals("receiver")){
            System.out.println("The searched feature is a receiver ");
            ReceiverExtractor receiver_extractor = new ReceiverExtractor();
            slice_class = receiver_extractor.extract_feature(feature, name_folder);
            if (slice_class != null) {
                slice_found = true;
                receiver_extractor.extractReceiver(feature);
            } else {
                System.out.println("Sorry, it was not possible to retrieve the intended feature from this application :( ");
                File dir = new File(output_dir);
                if (dir.isDirectory() && dir.exists()) {
                    try {
                        FileUtils.deleteDirectory(dir);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
                exit(0);
            }
        }else if(Feat_Type.equals("permission")){
            System.out.println("The searched feature is a permission ");
            PermissionExtractor permission_extractor = new PermissionExtractor();
            slice_class = permission_extractor.extract_feature(feature, name_folder);
            if (slice_class != null) {
                slice_found = true;
                permission_extractor.extractPermission(feature);
            } else {
                System.out.println("Sorry, it was not possible to retrieve the intended feature from this application :( ");
                File dir = new File(output_dir);
                if (dir.isDirectory() && dir.exists()) {
                    try {
                        FileUtils.deleteDirectory(dir);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
                exit(0);
            }
        }else if(Feat_Type.equals("providers")){
            System.out.println("The searched feature is a content provider ");
            ProviderExtractor provider_extractor = new ProviderExtractor();
            slice_class = provider_extractor.extract_feature(feature, name_folder);
            if (slice_class != null) {
                slice_found = true;
                provider_extractor.extractProvider(feature);
            } else {
                System.out.println("Sorry, it was not possible to retrieve the intended feature from this application :( ");
                File dir = new File(output_dir);
                if (dir.isDirectory() && dir.exists()) {
                    try {
                        FileUtils.deleteDirectory(dir);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
                exit(0);
            }
        }else if(Feat_Type.equals("intent")){
            System.out.println("The searched feature is a intent");
            IntentExtractor intent_extractor = new IntentExtractor();
            slice_class = intent_extractor.extract_feature(feature, name_folder);
            if (slice_class != null) {
                slice_found = true;
                intent_extractor.extractIntent(feature);
            } else {
                System.out.println("Sorry, it was not possible to retrieve the intended feature from this application :( ");
                File dir = new File(output_dir);
                if (dir.isDirectory() && dir.exists()) {
                    try {
                        FileUtils.deleteDirectory(dir);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
                exit(0);
            }
        }
        Options.v().set_output_format(Instrumenter.output_format);
        Options.v().set_output_dir(output_dir);
        if (new File(output_dir + "/classes.txt").exists()) {
            config.clean_final_export(output_dir + "/classes.txt");
            if (slice_class != null) {
                slice_class.setApplicationClass();
            }
            System.out.println("Now writing files into folder " + output_dir + " :) ");
            PackManager.v().writeOutput();
            if(slice_found){
                System.out.println("Dependencies exported and slice");
            }else{
                System.out.println("Dependencies exported but no slice");
            }
        } else if (new File(output_dir + "/class_of_extraction.txt").exists()) {
            config.clean_final_export(output_dir + "/class_of_extraction.txt");
            slice_class.setApplicationClass();
            PackManager.v().writeOutput();
            System.out.println("Slice exported");
        } else {
            System.out.println("Sorry, it was not possible to retrieve the intended feature from this application :( ");
            File dir = new File(output_dir);
            if(dir.isDirectory() && dir.exists()){
                try {
                    FileUtils.deleteDirectory(dir);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
        exit(0);

    }
}