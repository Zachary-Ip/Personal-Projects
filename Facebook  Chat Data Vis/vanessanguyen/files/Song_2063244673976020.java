// CSE 142 Spring 2019 Assignment 1
// 4-3-19, Zachary Ip
/////////////////////////////////////
public class Song {
  public static void main(String[] args){
  // First Verse
  swallowedFly();
  sheDie();
  // Second Verse
  swallowedSpider();
  spiderCatchFly();
  sheDie();
  // Third Verse
  swallowedBird();
  birdCatchSpider();
  spiderCatchFly();
  sheDie();
  // Fourth Verse
  swallowedCat();
  catCatchBird(); 
  birdCatchSpider();
  spiderCatchFly();
  sheDie();
  // Fifth Verse
  swallowedDog();
  dogCatchCat();
  catCatchBird(); 
  birdCatchSpider();
  spiderCatchFly();
  sheDie();
  // Sixth Verse
  swallowedTiger();
  tigerCatchDog();
  dogCatchCat();
  catCatchBird(); 
  birdCatchSpider();
  spiderCatchFly();
  sheDie();
  // Seventh Verse
  swallowedHorse();
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
  }
  public static void swallowedBird(){
    System.out.println("There was an old woman who swallowed a bird,");
    System.out.println("How absurd to swallow a bird.");
  }
  public static void birdCatchSpider(){
    System.out.println("She swallowed the bird to catch the spider,");
  }
  public static void swallowedCat(){
    System.out.println("There was an old woman who swallowed a cat,");
    System.out.println("Imagine that to swallow a cat.");
  }
  public static void catCatchBird(){
    System.out.println("She swallowed the cat to catch the bird,");
  }
  public static void swallowedDog(){
    System.out.println("There was an old woman who swallowed a dog,");
    System.out.println("What a hog to swallow a dog.");
  }
  public static void dogCatchCat(){
    System.out.println("She swallowed the dog to catch the cat,");
  }
  public static void swallowedTiger(){
    System.out.println("There was an old woman who swallowed a tiger,");
    System.out.println("She would need a lot of fiber to swallow a tiger.");
  }
  public static void tigerCatchDog(){
    System.out.println("She swallowed the tiger to catch the dog,");
  }
  public static void swallowedHorse(){
    System.out.println("There was an old woman who swallowed a horse,");
    System.out.println("She died of course.");
  }
}