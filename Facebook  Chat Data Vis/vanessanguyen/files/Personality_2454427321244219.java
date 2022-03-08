// Zachary Ip
// 5/24/19
// TA: Price Ludwig
// Assignment #7: Personality Test
//
// This program will
import java.util.*;
import java.io.*;

public class Personality {
   // int NUM_CLASSIFIERS defines how many catagories are being analyzed by questions
   public static final int NUM_CLASSIFIERS = 4;
   public static void main(String[] args) throws FileNotFoundException {
      intro();
      Scanner user = new Scanner(System.in);
      
      System.out.print("input file name? ");
      File file = new File(user.nextLine());
      
      System.out.print("output file name? ");
      PrintStream output = new PrintStream(user.nextLine());
      
      Scanner fileScan = new Scanner(file);
      int line = 1;
      
      // This loop runs through each of the entries in the file
      while(fileScan.hasNextLine()){
         if(line % 2 == 1) {
            output.print(fileScan.nextLine());
            output.print(": ");
         } else {
            String results = fileScan.nextLine();
            int[] percentB = count(results);
            String type = personalityType(percentB);
            output.print(Arrays.toString(percentB) + " = " + type + "\n");
         }
         line++;
      }  
   }
   
   // This method prints out an introduction to the user
   public static void intro() {
      System.out.println("This program processes a file of answers to the");
      System.out.println("Keirsey Temperament Sorter. It converts the");
      System.out.println("various A and B answers for each person into");
      System.out.println("a sequence of B-percentages and then into a");
      System.out.println("foud-letter personality type.");
      System.out.println();
   }
   
   // This method takes in a string that is the result for the personality test and scores it
   // String results - a single token that contains As and Bs case insennsitive 
   public static int[] count(String results) {
      // Creates arrays to hold the scores 
      int[] A = new int[NUM_CLASSIFIERS];
      int[] B = new int[NUM_CLASSIFIERS];
      
      // Creates an array to hold the final results
      int[] percentB = new int[NUM_CLASSIFIERS];
      for(int i = 0; i < results.length(); i++) {
         if(results.charAt(i) == 'A' || results.charAt(i) == 'a'){
            increase(A,i);
         } else if(results.charAt(i) == 'B' || results.charAt(i) == 'b') {
            increase(B,i);
         } else {
         }
      }
      for(int i = 0; i <=3; i++) {
         percentB[i] = (100 *B[i]) / (A[i] + B[i]);
      }
      return percentB;
   }
   
   // This method increases the score in the appropriate array at the appropriate location based
   // on which question was answered.
   // int[] array - sepcifies which score array to increase
   // int idx - determines which dimension of the test the question correlates to
   public static void increase(int[] array, int idx) {
      if(idx % 7 == 0) {
         array[0]++;
      } else if(idx % 7 == 1 || idx % 7 == 2) {
         array[1]++;
      } else if(idx % 7 == 3 || idx % 7 == 4) {
         array[2]++;
      } else {
         array[3]++;
      }
   }
   
   // This method takes in the final result array and determines what the personality type is
   // int[] array - final results array
   public static String personalityType(int[] array) {
      String A = "ESTJ";
      String B = "INFP";
      String type = "";
      for(int i = 0; i< NUM_CLASSIFIERS; i++) {
         if(array[i] < 50) {
            type += A.charAt(i);
         } else if(array[i] > 50) {
            type += B.charAt(i);
         } else {
            type += "X";
         }
      }
      return type;
   }
}



