// Zachary Ip
// 4/10/19
// TA: Price Ludwig
// Assignment #2: AsciiArt
//
// This program returns a spaceship design of variable size
public class AsciiArt {
   public static void main(String[] args) {
      int size = 3; // Define size of spaceship here
      tip(size);
      divider(size);
      bodyUp(size);
      bodyDown(size);
      divider(size);
      bodyDown(size);
      bodyUp(size);
      divider(size);
      tip(size);
   }
   // This method returns the tip or exhaust of the spaceship
   public static void tip(int tipSize) {
      for(int i = 0; i < tipSize*2-1; i++) {
         for(int j = tipSize*2-1-i; j > 0; j--) {
            System.out.print(" ");
         }
         for(int j = 0; j <= i; j++) {
            System.out.print("/");
         }
         System.out.print("**");
         for(int j = 0; j <= i; j++) {
            System.out.print("\\");
         }
         System.out.println();
      }
   }
   // This method returns a single line divider between sections of the ship
   public static void divider(int divSize) {
      System.out.print("+");
      for(int i = 0; i < divSize*2; i++) {
         System.out.print("=*");
      }
      System.out.print("+");
      System.out.println();
   }
   // This method returns a portion of the body of the spaceship with an ascending arrow
   public static void bodyUp(int upSize) {
      for (int i = 1; i <= upSize; i++) {
         System.out.print("|");
         dots(i, upSize);
         upArrow(i);
         dots(i, upSize);
         dots(i, upSize);
         upArrow(i);
         dots(i, upSize);
         System.out.print("|");
         System.out.println();
      }
   }
   // This method returns a portion of the body of the spaceship with a descending arrow
   public static void bodyDown(int downSize) {
      for (int i = downSize; i > 0; i--) {
         System.out.print("|");
         dots(i, downSize);
         downArrow(i);
         dots(i, downSize);
         dots(i, downSize);
         downArrow(i);
         dots(i, downSize);
         System.out.print("|");
         System.out.println();
      }
   }
   // This method returns a number of dots which properly spaces the # of arrows so for the body
   public static void dots(int dotSize, int max) {
      for (int j = max-dotSize; j > 0; j--) {
         System.out.print(".");
      }
   }
   // This method prints a number of upward arrows depending on input
   public static void upArrow(int arrowSize) {
      for (int j = 1; j <=arrowSize; j++) {
         System.out.print("/\\");
      }
   }
   // This method prints a number of downward arrows depending on input
   public static void downArrow(int arrowSize) {
      for (int j = 1; j <= arrowSize; j++) {
         System.out.print("\\/");
      }
   }
}