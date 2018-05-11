import java.util.*;
import java.lang.*;
public class train9
{
	static ArrayList<Integer> array;
  public static void main (String args[])
  {
  		array = new ArrayList<Integer>();
  		int fact = factorial(5);
  		
  		binaryConver(8);
  		for(int i = 0;i<array.size();i++){
  			System.out.println(array.get(i));
  		}
  		System.out.println("factorial is "+ fact +" multiplication is "+multiply);
  
} 
public static int factorial(int num){
	int fact = 1;

	for(int i = num;i> 0;i--){
		fact = fact * i;
	}
	return fact;
}
public static int multiply(int num1,int num2){
	return num1* num2;
}
public static int divide(int num1, int num2){
	return num1/num2;
}

public static void binaryConver(int num){
	if(num  <2 ){
		array.add(num);
	}
	else{
	binaryConver(num/2);
	array.add(num % 2);
}
}
public static int addNumbers(int num1 ,int num2){
	
	return num1 + num2;

}
public static int diffNumbers(int num1, int num2){
	return num2 - num1;
}

public static int power(int num1, int num2){
	int val = (int)Math.pow(num1,num2);
	return val;
} 
}


