# I'm just summarizing how this works so I can try and understand it.


## I'm defining some of the functions here

### /src/main.c

Processes all of the arguments. Creates a source based on the arguments given (libcamera, v4l2device,
slideshow...). This creates a video_source struct which is passed to some other funcions later.


#### /lib/configfs.c
struct uvc_function_config *configfs_parse_uvc_function(const char *function)


#### /lib/v4l2-source.c

struct video_source *v4l2_video_source_create(const char *devname)
This function is called when you are using the V4L2 device as a source
What this function does:

	1. Takes a string (devname) which is the path to the video src
	2. Allocates memory for a new v4l2_source which contains a video_source struct and v4l2_device 
	   struct
	3. Sets the ops field of the video_source to a defined video_source_ops struct
	4. Sets the type field of the video_source to VIDEO_SOURCE_DMABUF, this presumably means the source
	   uses DMA-BUF for memory management
	5. It tries to open the device, if this fails it frees the source from memory and returns null
	6. It checks the type of the v4l2_device to make sure it is V4L2_BUF_TYPE_VIDEO_CAPTURE, if not
	   it causes an error and returns.
	7. If everything is successful, it returns a pointer to src (the v4l2_source)


#### /lib/libcamera-source.cpp

stuct video_source *libcamera_source_create(const char *devname)
This function is called when you use libcamera as the source
What this function does:

	1. Takes a string (devname) which is the "camera identifier" (Usually this is just "0" which is the
	   index of the first camera)
	2. Creates a new libcamera_src object (which has a video_source struct, same as v4l2-source)
	3. Line 510: pipe2(src->pfds, O_NONBLOCK); described in the following paragraph

	This line of code is calling the pipe2 function to create a pipe, which is a mechanism for 
	interprocess communication (IPC) in Unix-like operating systems. A pipe provides a one-way 
	communication channel between two processes: one process writes data to the pipe, and the other 
	process reads that data.

	The pipe2 function takes two arguments:

	An array of two integers, src->pfds in this case, where the function will store the file 
	descriptors for the read end and write end of the pipe. After the function call, src->pfds[0] 
	will be the file descriptor for the read end of the pipe, and src->pfds[1] will be the file 
	descriptor for the write end.

	A set of flags that modify the behavior of the pipe. In this case, O_NONBLOCK is passed, which 
	means that the pipe is set to non-blocking mode. In non-blocking mode, read and write operations 
	on the pipe will return immediately rather than waiting for data to be available or for space to be 
	available for writing.

	The pipe2 function returns an integer, ret in this case, which is 0 on success and -1 on error. If 
	the function returns -1, an error occurred and the error code can be retrieved with errno.

	4. Sets the ops field of the video_source to a defined libcamera_source_ops struct
	5. Sets the type field of video_source to VIDEO_SOURCE_DMABUF
	6. Creates and starts a CameraManager object which is assigned to the libcamera_src (not sure what
	   the CameraManager object is/does)
	   edit: the CameraManager and other related objects are from the libcamera library. Look up
	   libcamera documentation.
	7. Finds the camera object through a somewhat complicated way of querying the cameras based on the
	   index you pass in (usually 0)
	8. Configures the camera through some generateConfiguration function that I need to figure out
	   what it does.
	9. Connects to the requestComplete method to the requestCompleted signal of the camera
	10. If everything is successful, returns a pointer to the src (libcamera_source)


After creating either the libcamera source or the v4l2 source, the program initializes the source.
This essentially binds the events field of the video_source.events


#### /lib/stream.c

struct uvc_stream *uvc_stream_new(const char *uvc_device)
This function is called and passed the uvc_device (your second argument, probably ucv.0)
What this function does:

	1. Creates a uvc_stream object and allocates memory to it.
	2. Calls uvc_open which calls v4l2_open, this contains some configuration that I will be looking
	   into.
	3. Returns pointer to uvc_stream object if everything is successful.


#### /lib/events.c

bool events_loop(struct events *events)
This is the main capture loop of uvc-gadget
What this function does:

	1. Sets events.done to false
	2. While events.done is false
	3. Creates the fs_set objects rfds, wfds, efds (not sure what these stand for)
	4. Runs some sort of system select command
	5. Dispatches an event as long as the system selct command is good.


## Here I am trying to modal all of the function that are called in main.c of uvc-gadget so that you can get an overview of what is going on. I am listing all of the main functionality and skipping some of the less important things. Although it is possible that I would miss something important.

```json
v4l2_video_source_create || libcamera_source_create
uvc_stream_new
	uvc_open
		v4l2_open
			* initializes memory for the v4l2 device
			* initializes the formats list
			* opens the v4l2 device using the open function with O_RDWR | O_NONBLOCK flags which means
				the device is opened for reading and writing and in non-blocking mode (so the open call 
				will return without waiting for the device to be ready)
			* queries the capabilities of the v4l2 device using the ioctl function with the
				VIDIOC_QUERYCAP command. It stores these. returns an error if the ioctl function is not
				successful
			* Checks the capabilities of the v4l2 device. If the device supports video capture, it
				sets the type member of the v4l2_device structure to V4L2_BUF_TYPE_VIDEO_CAPTURE. If
				the device supports video output, it sets the type member to V4L2_BUF_TYPE_VIDEO_OUTPUT.
				If the device supports neither, it prints an error message and closes the device and 
				returns null.
			v4l2_enum_formats
				* enumerates the formats supported by the v4l2 device
			* If all has been successful it prints a message that it has been successful
uvc_stream_set_event_handler
	* Sets the stream.events to the events passed in
uvc_stream_set_video_source
	* Sets stream.src to the src passed in
uvc_stream_init_uvc
	uvc_set_config
		* Sets uvc_device.fc to the fc passed in
	uvc_events_init
		uvc_fill_streaming_control(passes in probe)
		uvc_fill_streaming_control(passes in commit)
events_loop // main capture loop
cleanup functions
````


## Representation of some of the structures (basically objects) used


These are objects in main.c  
__src:__
```json
struct video_source {
	const struct video_source_ops *ops {
		void(*destroy)(struct video_source *src);
		int(*set_format)(struct video_source *src, struct v4l2_pix_format *fmt);
		int(*set_frame_rate)(struct video_source *src, unsigned int fps);
		int(*alloc_buffers)(struct video_source *src, unsigned int nbufs);
		int(*export_buffers)(struct video_source *src, struct video_buffer_set **buffers);
		int(*import_buffers)(struct video_source *src, struct video_buffer_set *buffers);
		int(*free_buffers)(struct video_source *src);
		int(*stream_on)(struct video_source *src);
		int(*stream_off)(struct video_source *src);
		int(*queue_buffer)(struct video_source *src, struct video_buffer *buf);
		void(*fill_buffer)(struct video_source *src, struct video_buffer *buf);
	},
	struct events *events {
		struct list_entry events {
			struct list_entry *prev
			struct list_entry *next
		}
		volatile bool done
		int maxfd
		fd_set rfds {
			__int32_t fds_bits
		}
		fd_set wfds;
		fd_set efds;
	}
	video_source_buffer_handler_t handler {
		void *
		struct video_source *
		struct video_buffer * {
			unsigned int index;
			unsigned int size;
			unsigned int bytesused;
			struct timeval timestamp;
			bool error;
			void *mem;
			int dmabuf;
		}
	}
	void *handler_data
	enum video_source_type type {
		VIDEO_SOURCE_DMABUF,
		VIDEO_SOURCE_STATIC,
		VIDEO_SOURCE_ENCODED,
	}
}
```

__stream:__
```json
struct uvc_stream {
	struct video_source *src {
		documented above
	}
	struct uvc_device *uvc {
		struct v4l2_device *vdev; {
			int fd;
			char *name;
			enum v4l2_buf_type type; {
				can't find this??
			}
			enum v4l2_memory memtype; {
				can't find this either??
			}
			struct list_entry formats; {
				struct list_entry *prev
				struct list_entry *next
			}
			struct v4l2_pix_format format; {
				something might be wrong i can't find this
			}
			struct v4l2_rect crop; {
				or this
			}
			unsigned int fps;
			struct video_buffer_set buffers; {
				struct video_buffer *buffers {
					unsigned int index;
					unsigned int size;
					unsigned int bytesused;
					struct timeval timestamp;
					bool error;
					void *mem;
					int dmabuf;
				}
				unsigned int nbufs
			}
		}
		struct uvc_stream *stream; {
			circular reference
		}
		struct uvc_function_config *fc; {
			char *video;
			char *udc;
			struct uvc_function_config_control control; {
				struct uvc_function_config_interface intf; {
					unsigned int bInterfaceNumber;
				}
			}
			struct uvc_function_config_streaming streaming; {
				struct uvc_function_config_interface intf; {
					unsigned int bInterfaceNumber;
				}
				struct uvc_function_config_endpoint ep; {
					unsigned int bInterval;
					unsigned int bMaxBurst;
					unsigned int wMaxPacketSize;
				}
				unsigned int num_formats;
				struct uvc_function_config_format *formats; {
					unsigned int index;
					uint8_t guid[16];
					unsigned int fcc;
					unsigned int num_frames;
					struct uvc_function_config_frame *frames; {
						unsigned int index;
						unsigned int width;
						unsigned int height;
						unsigned int num_intervals;
						unsigned int *intervals;
					}
				}
			}
		}
		struct uvc_streaming_control probe; {
			can't find this
		}
		struct uvc_streaming_control commit; {
			same as above (same struct)
		}
		int control;
		unsigned int fcc;
		unsigned int width;
		unsigned int height;
	}
	struct events *events {
		struct list_entry events {
			struct list_entry *prev
			struct list_entry *next
		}
		volatile bool done
		int maxfd
		fd_set rfds {
			__int32_t fds_bits
		}
		fd_set wfds;
		fd_set efds;
	}
}
```

__fc:__
```json
struct uvc_function_config *fc {
	char *video;
	char *udc;
	struct uvc_function_config_control control; {
		defined earlier
	}
	struct uvc_function_config_streaming streaming; {
		defined earlier
	}
}
```