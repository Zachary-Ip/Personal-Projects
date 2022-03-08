// CSE 142 Spring 2019 Assignment 1
// 4-3-19, Zachary Ip
/////////////////////////////////////
public class Song {
   public static void main(String[] args){
   // Verse 1
     sheSwallow("fly");
     sheDie();
   // Verse 2
     sheSwallow("spider");
     toCatch("spider","fly");
     sheDie();
   // Verse 3
     sheSwallow("bird");
     toCatch("bird", "spider","fly");
     sheDie();
   // Verse 4
     sheSwallow("cat");
     toCatch("cat", "bird", "spider","fly");
     sheDie();
   // Verse 5
     sheSwallow("dog");
     toCatch("dog","cat","bird", "spider","fly");
     sheDie();
   // Verse 6
     sheSwallow("tiger");
     toCatch("tiger","dog","cat","bird", "spider","fly");
     sheDie();
   //Verse 7
     sheSwallow("horse");
   }
   public static void sheSwallow(String x){
    if (x == "fly") {
      System.out.println("There was an old woman who swallowed a " + x + ".");
    } else {
    System.out.println("There was an old woman who swallowed a " + x + ",");
    }
      if (x == "spider") {
         System.out.println("That wriggled and iggled and jiggled inside her.");
      } else if(x == "bird") {
         System.out.println("How absurd to swallow a bird.");
       
      } else if(x == "cat") {
         System.out.println("Imagine that to swallow a cat.");
       
      } else if(x == "dog") {
         System.out.println("What a hog to swallow a dog.");
       
      } else if(x == "tiger") {
         System.out.println("You need some fiber to swallow a tiger.");
       
      } else if(x == "horse") {
         System.out.println("She died of course.");
      }
   }
   public static void sheDie(){
      System.out.println("I don't know why she swallowed that fly,");
      System.out.println("Perhaps she'll die.");
      System.out.println();
   }
     
   public static void toCatch(String...args) {
      System.out.println("She swallowed the " + args[0] + " to catch the " + args[1] + ",");
      if (args.length > 2){
         System.out.println("She swallowed the " + args[1] + " to catch the " + args[2] + ",");
      }
      if (args.length > 3){
         System.out.println("She swallowed the " + args[2] + " to catch the " + args[3] + ",");
      }
      if (args.length > 4){
         System.out.println("She swallowed the " + args[3] + " to catch the " + args[4] + ",");
      }
      if (args.length > 5) {
         System.out.println("She swallowed the " + args[4] + " to catch the " + args[5] + ",");
      }
   }
}