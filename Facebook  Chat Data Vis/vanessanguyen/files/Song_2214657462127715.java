// CSE 142 Spring 2019 Assignment 1
// Zachary Ip, 4/3/19
public class Song {
  public static void main(String[] args){
    swallowedFly();  // First Verse
    sheDie();
 
    swallowedSpider();  // Second Verse
    spiderCatchFly();
 
    swallowedBird();  // Third Verse
    birdCatchSpider();
    
    swallowedCat(); // Fourth Verse
    catCatchBird(); 
  
    swallowedDog();  // Fifth Verse
    dogCatchCat();
  
    swallowedTiger(); // Sixth Verse
    tigerCatchDog();
 
    swallowedHorse();  // Seventh Verse
  }
  public static void swallowedFly(){
    System.out.println("There was an old woman who swallowed a fly.");
  }
  public static void sheDie(){
    System.out.println("I don't know why she swallowed that fly,");
    System.out.println("Perhaps she'll die.");
    System.out.println();
  }
  public static void swallowedSpider(){
    System.out.println("There was an old woman who swallowed a spider,");
    System.out.println("That wriggled and iggled and jiggled inside her.");
  }
  public static void spiderCatchFly(){
    System.out.println("She swallowed the spider to catch the fly,");
    sheDie();
  }
  public static void swallowedBird(){
    System.out.println("There was an old woman who swallowed a bird,");
    System.out.println("How absurd to swallow a bird.");
  }
  public static void birdCatchSpider(){
    System.out.println("She swallowed the bird to catch the spider,");
    spiderCatchFly();
  }
  public static void swallowedCat(){
    System.out.println("There was an old woman who swallowed a cat,");
    System.out.println("Imagine that to swallow a cat.");
  }
  public static void catCatchBird(){
    System.out.println("She swallowed the cat to catch the bird,");
    birdCatchSpider();
  }
  public static void swallowedDog(){
    System.out.println("There was an old woman who swallowed a dog,");
    System.out.println("What a hog to swallow a dog.");
  }
  public static void dogCatchCat(){
    System.out.println("She swallowed the dog to catch the cat,");
    catCatchBird();
  }
  public static void swallowedTiger(){
    System.out.println("There was an old woman who swallowed a tiger,");
    System.out.println("She would need a lot of fiber to swallow a tiger.");
  }
  public static void tigerCatchDog(){
    System.out.println("She swallowed the tiger to catch the dog,");
    dogCatchCat();
  }
  public static void swallowedHorse(){
    System.out.println("There was an old woman who swallowed a horse,");
    System.out.println("She died of course.");
  }
}