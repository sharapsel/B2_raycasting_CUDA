#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <fstream>
#include <cmath>
#include <png.h>
#include <time.h>
#include <iostream>
#include <iomanip>
#include <thrust\host_vector.h>
#include <thrust\device_vector.h>
#define SPHERE_COUNT 200
using namespace std;
using namespace thrust;

struct Vec3
{
	double x, y, z;
	__device__ __host__ Vec3() {}
	__device__ __host__ Vec3(double x, double y, double z) : x(x), y(y), z(z) {}
	__device__ __host__ Vec3 operator + (Vec3& v)
	{
		return Vec3(x + v.x, y + v.y, z + v.z);
	}
	__device__ __host__ Vec3 operator - (Vec3& v)
	{
		return Vec3(x - v.x, y - v.y, z - v.z);
	}
	__device__ __host__ Vec3 operator * (double d)
	{
		return Vec3(x * d, y * d, z * d);
	}
	__device__ __host__ Vec3 operator * (Vec3& v)
	{
		return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
	}
	__device__ __host__ Vec3 operator / (double d)
	{
		return Vec3(x / d, y / d, z / d);
	}
	__device__ __host__ Vec3 normalize()
	{
		double n = sqrt(x * x + y * y + z * z);
		return Vec3(x / n, y / n, z / n);
	}
};

__device__ __host__ double dot(Vec3& a, Vec3& b)
{
	return (a.x * b.x + a.y * b.y + a.z * b.z);
}

struct Ray
{
	Vec3 o, d;
	__device__ __host__ Ray(Vec3& o, Vec3& d) : o(o), d(d) {}
};

struct Sphere
{
	Vec3 c;
	double r;
	Vec3 col;
	__device__ __host__ Sphere() {}
	__device__ __host__ Sphere(Vec3& c, double r, Vec3& col) : c(c), r(r), col(col) {}
	__device__ __host__ Vec3 getNormal(Vec3& pi)
	{
		return (pi - c) / r;
	}
	__device__ __host__ bool intersect(Ray& ray, double &t)
	{
		Vec3 o = ray.o;
		Vec3 d = ray.d;
		Vec3 oc = o - c;
		double b = 2 * dot(oc, d);
		double c = dot(oc, oc) - r*r;
		double disc = b * b - 4 * c;
		if (disc < 1e-4)
			return false;
		disc = sqrt(disc);
		double t0 = -b - disc;
		double t1 = -b + disc;
		t = (t0 < t1) ? t0 : t1;
		return true;
	}
	__device__ __host__ bool sphereIntersect(Sphere& s2)
	{
		Vec3 delta = s2.c - c;
		double dist = sqrt(dot(delta, delta));
		return dist <= r + s2.r;
	}
};

struct Pixel
{
	uint8_t r;
	uint8_t g;
	uint8_t b;
	__device__ __host__ Pixel() : r(0), g(0), b(0) {}
	__device__ __host__  Pixel(uint8_t r, uint8_t g, uint8_t b) : r(r), g(g), b(b) {}
};

__device__ __host__ void vecToUINT8(Vec3& col, Pixel& p)
{
	p.r = (uint8_t)((col.x > 255) ? 255 : (col.x < 0) ? 0 : col.x);
	p.g = (uint8_t)((col.y > 255) ? 255 : (col.y < 0) ? 0 : col.y);
	p.b = (uint8_t)((col.z > 255) ? 255 : (col.z < 0) ? 0 : col.z);
}

struct Bitmap
{
	int h, w;
	Pixel* pixels;
	__host__ Bitmap(int h, int w) : h(h), w(w)
	{
		pixels = new Pixel[h * w];
	}
	__host__ void Clear()
	{
		for (int i = 0; i < h*w; i++)
			pixels[i] = Pixel();
	}
	__host__ ~Bitmap()
	{
		delete(pixels);
	}
};

__host__ int saveBitmapToPNG(Bitmap &b, const char *path)
{
	/* create file */
	FILE * fp;
	png_structp png_ptr = NULL;
	png_infop info_ptr = NULL;
	int pixel_size = 3;
	int depth = 8;
	int code = 0;
	fp = fopen(path, "wb");
	if (!fp)
	{
		code = 1;
		goto finalise;
	}
	/* initialize stuff */
	png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (png_ptr == NULL)
	{
		code = 2;
		goto finalise;
	}
	info_ptr = png_create_info_struct(png_ptr);
	if (info_ptr == NULL)
	{
		code = 3;
		goto finalise;
	}
	if (setjmp(png_jmpbuf(png_ptr)))
	{
		code = 4;
		goto finalise;
	}
	png_init_io(png_ptr, fp);
	/* write header */
	if (setjmp(png_jmpbuf(png_ptr)))
	{
		code = 5;
		goto finalise;
	}
	png_set_IHDR(png_ptr, info_ptr, b.w, b.h, depth, PNG_COLOR_TYPE_RGB,
		PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
	png_write_info(png_ptr, info_ptr);
	/* write bytes */
	if (setjmp(png_jmpbuf(png_ptr)))
	{
		code = 6;
		goto finalise;
	}
	png_bytep row = (png_bytep)malloc(pixel_size * b.w * sizeof(png_byte));
	int rev, revCoord;
	for (int y = 0; y < b.h; y++)
	{
		for (int x = 0; x < b.w; x++)
		{
			png_byte* ptr = &(row[x * 3]);
			rev = b.h - y - 1;
			revCoord = rev * b.w + x;
			ptr[0] = b.pixels[revCoord].r;
			ptr[1] = b.pixels[revCoord].g;
			ptr[2] = b.pixels[revCoord].b;
		}
		png_write_row(png_ptr, row);
	}
	/* end write */
	if (setjmp(png_jmpbuf(png_ptr)))
	{
		code = 7;
		goto finalise;
	}
	png_write_end(png_ptr, NULL);
finalise:
	if (fp != NULL) fclose(fp);
	if (info_ptr != NULL) png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
	if (png_ptr != NULL) png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
	if (row != NULL) delete(row);
	return code;
}

__host__ void generateSpheres(host_vector<Sphere> &spheres, int h, int w, int sphereCount)
{
	srand(123456);
	spheres = host_vector<Sphere>(sphereCount);
	for (int i = 0; i < sphereCount; i++)
	{
		Sphere s(
			Vec3(rand() % w, rand() % h, rand() % h / 2),
			h / 20 + (rand() % h / 10),
			Vec3(rand() % 256, rand() % 256, rand() % 256));
		bool isOk = true;
		for (int j = 0; j < i; j++)
			if (spheres[j].sphereIntersect(s))
			{
				isOk = false;
				break;
			}
		if (isOk)
			spheres[i] = s;
		else i--;
	}
}

__host__ bool intersectArray(host_vector<Sphere> &spheres, Ray &ray, double &t, int &id, int spheresCount)
{
	bool intersects = false;
	double d = 0;
	t = 1e10;
	for (int i = 0; i < spheresCount; i++)
	{
		intersects = spheres[i].intersect(ray, d);
		if (intersects && d < t)
		{
			t = d;
			id = i;
		}
	}
	return t < 1e10;
}

__device__ bool intersectArray(Sphere *spheres, Ray ray, double &t, int &id, int spheresCount)
{
	bool intersects = false;
	double d = 0;
	t = 1e10;
	for (int i = 0; i < spheresCount; i++)
	{
		intersects = spheres[i].intersect(ray, d);
		if (intersects && d < t)
		{
			t = d;
			id = i;
		}
	}
	return t < 1e10;
}

__host__ void raytraceHost(Bitmap &b, Sphere *light, host_vector<Sphere> &spheres, int spheresCount)
{
	Vec3 black(0, 0, 0);
#pragma omp parallel for
	for (int y = 0; y < b.h; ++y)
	{
		for (int x = 0; x < b.w; ++x)
		{
			Vec3 pix_col(black);
			double t;
			int id;
			Ray ray(Vec3(x, y, -100), Vec3(0, 0, 1));
			bool intersects = intersectArray(spheres, ray, t, id, spheresCount);
			if (intersects)
			{
				Vec3 crossP = ray.o + ray.d*t;
				Vec3 lightDir = (*light).c;
				//Vec3 lightDir = (*light).c - crossP;
				Vec3 normal = spheres[id].getNormal(crossP);
				double lambertian = dot(lightDir.normalize(), normal.normalize());
				pix_col = (spheres[id].col * lambertian) * 0.7;
				//Vec3 specColor(0.1, 0.1, 0.1);
				vecToUINT8(pix_col, b.pixels[y * b.w + x]);
			}
		}
	}
}

__global__ void raytraceKernel(Pixel *pixels, Sphere *light, Sphere *spheres, int h, int w, int spheresCount)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w || y >= h)
		return;
	Vec3 pix_col(0,0,0);
	double t;
	int id;
	Ray ray(Vec3(x, y, -100), Vec3(0, 0, 1));
	bool intersects = intersectArray(spheres, ray, t, id, spheresCount);
	if (intersects)
	{
		Vec3 crossP = ray.o + ray.d*t;
		Vec3 lightDir = (*light).c;
		//Vec3 lightDir = (*light).c - crossP;
		Vec3 normal = spheres[id].getNormal(crossP);
		double lambertian = max(dot(lightDir.normalize(), normal.normalize()), 0.0);
		pix_col = (spheres[id].col * lambertian) * 0.7;
		//Vec3 specColor(0.1, 0.1, 0.1);
		//if (lambertian > 0.000001)
		//{
		//	Vec3 viewDir = crossP.normalize() * (-1);
		//	Vec3 halfDir = (lightDir + viewDir).normalize();
		//	double specAngle = max(dot(halfDir, normal), 0.0);
		//	double specular = max(dot(normal, lightDir), 0.0) * pow(specAngle, 60);
		//	pix_col = pix_col + Vec3(0.2, 0.2, 0.2) * specular;
		//}
		vecToUINT8(pix_col, pixels[y * w + x]);
	}
}

void raytraceCuda(Pixel * pixels, Sphere *light, host_vector<Sphere> &spheres, int h, int w, int spheresCount)
{
	cudaError_t cudaStatus = cudaSetDevice(0);
	//pixels
	Pixel *p_pixels;
	cudaMalloc((void**)&p_pixels, w * h * sizeof(Pixel));
	cudaMemcpy(p_pixels, pixels, w * h * sizeof(Pixel), cudaMemcpyHostToDevice);
	//light
	Sphere *p_light;
	cudaMalloc((void**)&p_light, sizeof(Sphere));
	cudaMemcpy(p_light, light, sizeof(Sphere), cudaMemcpyHostToDevice);
	//spheres
	device_vector<Sphere> dev_spheres = spheres;
	Sphere *p_spheres = raw_pointer_cast(dev_spheres.data());
	//kernel
	dim3 dimBlock(8, 8);
	int xBlocks = w / dimBlock.x + ((w % dimBlock.x) == 0 ? 0 : 1);
	int yBlocks = h / dimBlock.y + ((h % dimBlock.y) == 0 ? 0 : 1);
	dim3 dimGrid = dim3(xBlocks, yBlocks);
	raytraceKernel << < dimGrid, dimBlock >> > (p_pixels, p_light, p_spheres, h, w, spheresCount);
	//to host
	cudaMemcpy(pixels, p_pixels, w * h * sizeof(Pixel), cudaMemcpyDeviceToHost);
	return;
}

int main()
{
	clock_t tStart = clock();
	host_vector<Sphere> spheres;
	int h[] = { 720, 900, 1080, 1440};
	int w[] = { 1280, 1600, 1920, 2560 };
	int sphereCount[] = { 100, 200, 300, 400 };
	Sphere lights[] = {
		Sphere(Vec3(200, 500, 150), 1, Vec3(255, 255, 255)),
		Sphere(Vec3(500, 600, 150), 1, Vec3(255, 255, 255)),
		Sphere(Vec3(1300, 200, 200), 1, Vec3(255, 255, 255)),
		Sphere(Vec3(2500, 1400, -100), 1, Vec3(255, 255, 255))
	};
	for (int i = 0; i < 4; i++)
	{
		spheres.clear();
		generateSpheres(spheres, h[i], w[i], sphereCount[i]);
		Bitmap b(h[i], w[i]);

		//HOST
		tStart = clock();
		raytraceHost(b, &lights[i], spheres, sphereCount[i]);
		cout << "HOST - time elapsed: " << setprecision(5) << (double)(clock() - tStart) / CLOCKS_PER_SEC << " s" << endl;
		string hS = "file_host_";
		hS.operator+=(to_string(i)).operator+=(".png");
		saveBitmapToPNG(b, hS.c_str());
		//CUDA
		b.Clear();
		tStart = clock();
		try
		{
			raytraceCuda(b.pixels, &lights[i], spheres, b.h, b.w, sphereCount[i]);
		}
		catch (exception){}
		cout << "CUDA - time elapsed: " << setprecision(5) << (double)(clock() - tStart) / CLOCKS_PER_SEC << " s" << endl;
		string cS = "file_cuda_";
		cS.operator+=(to_string(i)).operator+=(".png");
		saveBitmapToPNG(b, cS.c_str());
	}
	
    cudaDeviceReset();
	getchar();
    return 0;
}