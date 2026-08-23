import math
import cv2
import numpy as np
import yaml
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

# Load YAML Configurations
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

app = FastAPI(title=config["app"]["name"])


class VectorBallProjection:

    def __init__(self, radius, rings, segments):
        self.radius = radius
        self.rings = rings
        self.segments = segments
        self.angle_x = 0.0
        self.angle_y = 0.0
        self.vertices, self.edges = self._generate_sphere()

    def _generate_sphere(self):
        vertices = []
        edges = []

        for i in range(self.rings + 1):
            theta = i * math.pi / self.rings
            sin_theta = math.sin(theta)
            cos_theta = math.cos(theta)

            for j in range(self.segments + 1):
                phi = j * 2 * math.pi / self.segments
                x = self.radius * sin_theta * math.cos(phi)
                y = self.radius * cos_theta
                z = self.radius * sin_theta * math.sin(phi)
                vertices.append([x, y, z])

        for i in range(self.rings):
            for j in range(self.segments):
                first = i * (self.segments + 1) + j
                second = first + self.segments + 1
                edges.append((first, second))
                edges.append((first, first + 1))

        return np.array(vertices, dtype=np.float32), edges

    def project_and_draw(
        self,
        frame,
        rot_speed_x,
        rot_speed_y,
        line_color,
        point_color,
        thickness,
    ):
        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2

        self.angle_x += rot_speed_x
        self.angle_y += rot_speed_y

        # Rotation Matrices
        rx = np.array(
            [
                [1, 0, 0],
                [0, math.cos(self.angle_x), -math.sin(self.angle_x)],
                [0, math.sin(self.angle_x), math.cos(self.angle_x)],
            ]
        )

        ry = np.array(
            [
                [math.cos(self.angle_y), 0, math.sin(self.angle_y)],
                [0, 1, 0],
                [-math.sin(self.angle_y), 0, math.cos(self.angle_y)],
            ]
        )

        rotated = np.dot(self.vertices, rx.T)
        rotated = np.dot(rotated, ry.T)

        # Perspective Projection
        distance = 400
        projected_points = []
        for point in rotated:
            z = point[2] + distance
            fov = distance / z if z != 0 else 1.0
            x_proj = int(point[0] * fov + cx)
            y_proj = int(point[1] * fov + cy)
            projected_points.append((x_proj, y_proj))

        # Render Edges
        for p1_idx, p2_idx in self.edges:
            p1 = projected_points[p1_idx]
            p2 = projected_points[p2_idx]
            cv2.line(frame, p1, p2, line_color, thickness, cv2.LINE_AA)

        # Render Vertices
        for p in projected_points:
            cv2.circle(frame, p, 2, point_color, -1, cv2.LINE_AA)

        return frame


proj = VectorBallProjection(
    radius=config["projection"]["radius"],
    rings=config["projection"]["rings"],
    segments=config["projection"]["segments"],
)


def generate_camera_stream():
    cap = cv2.VideoCapture(config["camera"]["device_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["camera"]["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera"]["frame_height"])

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)  # Mirror frame
        processed_frame = proj.project_and_draw(
            frame,
            rot_speed_x=config["projection"]["rotation_speed_x"],
            rot_speed_y=config["projection"]["rotation_speed_y"],
            line_color=tuple(config["projection"]["wireframe_color"]),
            point_color=tuple(config["projection"]["vertex_color"]),
            thickness=config["projection"]["line_thickness"],
        )

        _, buffer = cv2.imencode(".jpg", processed_frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )

    cap.release()


@app.get("/")
def index():
    return HTMLResponse(
        """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Vector Stream UI</title>
        <style>
            body { background: #0f172a; color: white; text-align: center; font-family: sans-serif; margin: 0; padding: 20px; }
            .feed { border: 2px solid #38bdf8; border-radius: 8px; margin-top: 15px; width: 640px; height: 480px; }
        </style>
    </head>
    <body>
        <h2>Live Physical Vector Projection Engine</h2>
        <img class="feed" src="/video_feed" />
    </body>
    </html>
    """
    )


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_camera_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app, host=config["app"]["host"], port=int(config["app"]["port"])
    )
