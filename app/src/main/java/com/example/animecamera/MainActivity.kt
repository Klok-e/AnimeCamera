package com.example.animecamera

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.media.ExifInterface
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.core.graphics.get
import androidx.core.graphics.scale
import androidx.core.graphics.set
import kotlinx.android.synthetic.main.activity_main.*
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.BuildConfig
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.lang.Exception
import java.nio.ByteBuffer
import java.nio.IntBuffer
import java.text.SimpleDateFormat
import java.util.*
import kotlin.experimental.and


class MainActivity : AppCompatActivity() {
    val REQUEST_IMAGE_CAPTURE = 1
    val channels = 3

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
    }

    fun takePicture(_view: View) {
        Intent(MediaStore.ACTION_IMAGE_CAPTURE).also { takePictureIntent ->
            takePictureIntent.resolveActivity(packageManager)?.also {
                // Create the File where the photo should go
                val photoFile: File? = try {
                    createImageFile()
                } catch (ex: IOException) {
                    // Error occurred while creating the File
                    null
                }
                // Continue only if the File was successfully created
                photoFile?.also {
                    val photoURI = FileProvider.getUriForFile(
                        this,
                        "com.example.animecamera",
                        it
                    )
                    takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI)
                    startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
                }
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            val imageBitmap =
                applyTransformation(rotateBasedOnStuff(BitmapFactory.decodeFile(currentPhotoPath)))
            pictureView.setImageBitmap(imageBitmap)
        }
    }

    lateinit var currentPhotoPath: String

    private fun createImageFile(): File {
        // Create an image file name
        val timeStamp: String =
            SimpleDateFormat("yyyy_MM_dd_HH_mm_ss", Locale.ENGLISH).format(Date())
        val storageDir: File = getExternalFilesDir("images")!!
        return File.createTempFile(
            "JPEG_${timeStamp}_", /* prefix */
            ".jpg", /* suffix */
            storageDir /* directory */
        ).apply {
            // Save a file: path for use with ACTION_VIEW intents
            currentPhotoPath = absolutePath
        }
    }

    /**
     * Copies specified asset to the file in /files app directory and returns this file absolute path.
     *
     * @return absolute file path
     */
    @Throws(IOException::class)
    fun assetFilePath(assetName: String): String? {
        val file = File(filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }
        assets.open(assetName).use { stream ->
            FileOutputStream(file).use { os ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (stream.read(buffer).also { read = it } != -1) {
                    os.write(buffer, 0, read)
                }
                os.flush()
            }
            return file.absolutePath
        }
    }

    private fun rotateBasedOnStuff(bitmap: Bitmap):Bitmap {
        val ei = ExifInterface(currentPhotoPath)
        val orientation = ei.getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_UNDEFINED
        )

        val rotatedBitmap: Bitmap =
            when (orientation) {

                ExifInterface.ORIENTATION_ROTATE_90 -> rotateBitmap(bitmap, 90f);
                ExifInterface.ORIENTATION_ROTATE_180 -> rotateBitmap(bitmap, 180f);
                ExifInterface.ORIENTATION_ROTATE_270 -> rotateBitmap(bitmap, 270f);
                else -> bitmap;
            }
        return rotatedBitmap
    }

    private fun rotateBitmap(source: Bitmap, angle: Float): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(angle)
        return Bitmap.createBitmap(
            source,
            0,
            0,
            source.width,
            source.height,
            matrix,
            true
        )
    }

    private fun applyTransformation(image: Bitmap): Bitmap {
        val module: Module = Module.load(assetFilePath("generator_A-B_2.pt"))
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            image.scale(240, 240),
            arrayOf(0.5f, 0.5f, 0.5f).toFloatArray(),
            arrayOf(0.5f, 0.5f, 0.5f).toFloatArray()
        )
        val output = module.forward(IValue.from(inputTensor)).toTensor()
        return floatArrayToBitmap(output.dataAsFloatArray, 240, 240)
            .scale(pictureView.width, pictureView.height)
    }

    private fun floatArrayToBitmap(
        floatArray: FloatArray,
        width: Int,
        height: Int
    ): Bitmap {
        fun convert(x: Float): Int {
            return ((x + 1f) / 2f * 255f).toInt()
        }

        val pixels = width * height;
        val offsetr = 0
        val offsetg = pixels
        val offsetb = 2 * pixels
        val byteBuffer = IntBuffer.allocate(width * height)
        // copy each value from float array to RGB channels and set alpha channel
        for (i in 0 until pixels) {
            val R = convert(floatArray[i + offsetr])
            val G = convert(floatArray[i + offsetg])
            val B = convert(floatArray[i + offsetb])
            val A = 255

            //int color = (A & 0xff) << 24 | (B & 0xff) << 16 | (G & 0xff) << 8 | (R & 0xff);
            val color: Int =
                ((A and 0xff) shl 24) or ((B and 0xff) shl 16) or ((G and 0xff) shl 8) or (R and 0xff)
            byteBuffer.put(color)
        }

        // Create empty bitmap in RGBA format (even though it says ARGB but channels are RGBA)
        val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        byteBuffer.rewind()
        bmp.copyPixelsFromBuffer(byteBuffer)

        return bmp
    }
}
