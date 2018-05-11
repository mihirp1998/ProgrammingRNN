import java.util.*;
import java.lang.*;
public class train1
{
  static ArrayList<Integer> array;
  public static void main (String args[])
  {
      array = new ArrayList<Integer>();
      int fact = factorial(5);
      int multiply = multiply(5,10);
      binaryConver(8);
      for(int i = 0;i<array.size();i++){
        System.out.println(array.get(i));
      }
      System.out.println("factorial is "+ fact );
      int val = power(3,4);
      System.out.println("power val is "+ val);
} 
public static int factorial(int num){
  int fact = 1;

  for(int i = num;i> 0;i--){
    fact = fact * i;
  }
  return fact;
}
 
}


