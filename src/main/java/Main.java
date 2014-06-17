import org.bytedeco.javacpp.opencv_core.*;
import org.bytedeco.javacpp.opencv_objdetect.CvHaarClassifierCascade;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_highgui.cvLoadImage;
import static org.bytedeco.javacpp.opencv_highgui.cvShowImage;
import static org.bytedeco.javacpp.opencv_highgui.cvWaitKey;
import static org.bytedeco.javacpp.opencv_objdetect.cvHaarDetectObjects;

public class Main {

    public static final String XML_FILE =
            "resources/haarcascade_frontalface_default.xml";

    public static void main(String[] args) {
        IplImage img = cvLoadImage("resources/lena.png");
        detect(img);
    }

    public static void detect(IplImage src){
        CvHaarClassifierCascade cascade = new
                CvHaarClassifierCascade(cvLoad(XML_FILE));
        CvMemStorage storage = CvMemStorage.create();
        System.out.println("Before cvHaarDetectObjects");
        CvSeq sign = cvHaarDetectObjects(
                src,
                cascade,
                storage,
                1.5,
                3,
                0);
        System.out.println("After cvHaarDetectObjects");

        cvClearMemStorage(storage);

        int total_Faces = sign.total();

        for(int i = 0; i < total_Faces; i++){
            CvRect r = new CvRect(cvGetSeqElem(sign, i));
            cvRectangle (
                    src,
                    cvPoint(r.x(), r.y()),
                    cvPoint(r.width() + r.x(), r.height() + r.y()),
                    CvScalar.RED,
                    2,
                    CV_AA,
                    0);

        }

        cvShowImage("Result", src);
        cvWaitKey(0);
    }
}
