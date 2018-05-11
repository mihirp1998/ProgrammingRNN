import java.net.*;
import java.io.*;
public class image
{
    public static void main(String[] args) throws Exception
    {
       //open a socket to the URL www.java2s.com"
        InetAddress IPobject=InetAddress.getByName("www.bits-pilani.ac.in");
        Socket Fromclient= new Socket(IPobject,80);
                                                                                                                                                                                                     PrintWriter OutToServer= new PrintWriter(Fromclient.getOutputStream());
       OutToServer.println("GET  //Uploads/Campus/BITS_Dubai_campus_logo.gif  HTTP/1.1");
       OutToServer.println("Host: www.bits-.pilani.ac.in");
       OutToServer.println("");//send a blank line 
       OutToServer.flush();//do not forget 
      InputStream InputFromServ= Fromclient.getInputStream();// attach an inputstream to socket
      File Fileobj= new File("./Image.gif");
      FileOutputStream OutToFile= new FileOutputStream(Fileobj);
      byte[] ImageBytes= new byte[2048];
      //read block of data via inputstream and put it into the defined byte array
      int length;
      boolean HeadersComplete=false;
      while((length=InputFromServ.read(ImageBytes))!=-1)
      
      {
          
              if(HeadersComplete)// already got rid of all the headers
              {
                  OutToFile.write(ImageBytes,0,length);//for further image data from server
                  //after /r/n/r/n detected
                }
                else// still yet to get rid of end of headers 
                {
                    for(int i=0;i< 2045;i++)
                    {
                    if(ImageBytes[i]==13 && ImageBytes[i+1]==10 && ImageBytes[i+2]==13 && ImageBytes[i+3]==10)
                    {
                      HeadersComplete=true;
                      OutToFile.write(ImageBytes,i+4,2048-i-4);
                      break;// first time end of headers detected
                    }
                   
                }//end of for loop
            }// end of else
          
      //OutToFile.write(ImageBytes,0,length);
    }// end of while
    OutToFile.flush();//flush the fileoutputstream to push all information
    // Fileobj.close();   
        
    }
    
}
