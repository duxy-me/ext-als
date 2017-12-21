package utils;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import data_structure.DenseMatrix;

public class LatentFactorUtil {
	public static void save(String filename, DenseMatrix U, DenseMatrix V) throws IOException {
		FileOutputStream fout = new FileOutputStream(filename);
		ObjectOutputStream oos = new ObjectOutputStream(fout);
		oos.writeObject(U);
		oos.writeObject(V);
	}

	public static DenseMatrix[] load(String filename) {
		FileInputStream streamIn;
		try {
			streamIn = new FileInputStream(filename);
			ObjectInputStream objectinputstream = new ObjectInputStream(streamIn);
			DenseMatrix U = (DenseMatrix) objectinputstream.readObject();
			DenseMatrix V = (DenseMatrix) objectinputstream.readObject();
			//System.out.println(U.mult(V.transpose()));
			return new DenseMatrix[] {U,V};
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}
}
