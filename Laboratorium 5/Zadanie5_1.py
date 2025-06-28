import numpy as np
from matplotlib import pyplot as plt

# Ustawienie rozmiaru wyswietlanego obrazu
plt.rcParams['figure.figsize'] = [12, 8]
EPSILON = 0.0001 # Mala wartosc do eliminacji bledow numerycznych

# Funkcja zwracajaca wektor odbity wzgledem normalnej
def reflect(vector, normal_vector):
    n_dot_l = np.dot(vector, normal_vector)
    return vector - normal_vector * (2 * n_dot_l)

# Funkcja zwracajaca znormalizowany wektor (dlugosc = 1)
def normalize(vector):
    return vector / np.sqrt((vector**2).sum())

# Klasa reprezentujaca promien
class Ray:
    def __init__(self, starting_point, direction):
        self.starting_point = starting_point
        self.direction = direction

# Klasa reprezentujaca swiatlo w scenie
class Light:
    def __init__(self, position):
        self.position = position
        self.ambient = np.array([0, 0, 0])
        self.diffuse = np.array([0, 1, 1])
        self.specular = np.array([1, 1, 0])

# Bazowa klasa dla wszystkich obiektow w scenie
class SceneObject:
    def __init__(self, ambient=np.array([0, 0, 0]),
                 diffuse=np.array([0.6, 0.7, 0.8]),
                 specular=np.array([0.8, 0.8, 0.8]),
                 shining=25):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shining = shining

    # Funkcja zwracajaca normalna w punkcie przeciecia
    def get_normal(self, cross_point):
        raise NotImplementedError

    # Funkcja sprawdzajaca przeciecie promienia z obiektem
    def trace(self, ray):
        raise NotImplementedError

    # Funkcja obliczajaca kolor w punkcie przeciecia promienia z obiektem
    def get_color(self, cross_point, obs_vector, scene):
        color = self.ambient * scene.ambient
        light = scene.light

        normal = self.get_normal(cross_point)
        light_vector = normalize(light.position - cross_point)
        n_dot_l = np.dot(light_vector, normal)
        reflection_vector = normalize(reflect(-1 * light_vector, normal))
        v_dot_r = np.dot(reflection_vector, -obs_vector)

        if v_dot_r < 0:
            v_dot_r = 0

        if n_dot_l > 0:
            color += (
                (self.diffuse * light.diffuse * n_dot_l) +
                (self.specular * light.specular * v_dot_r ** self.shining) +
                (self.ambient * light.ambient)
            )

        return color

# Klasa reprezentujaca kule
class Sphere(SceneObject):
    def __init__(self, position, radius,
                 ambient=np.array([0, 0, 0]),
                 diffuse=np.array([0.6, 0.7, 0.8]),
                 specular=np.array([0.8, 0.8, 0.8]),
                 shining=25):
        super(Sphere, self).__init__(ambient, diffuse, specular, shining)
        self.position = position
        self.radius = radius

    # Zwraca normalna w punkcie przeciecia
    def get_normal(self, cross_point):
        return normalize(cross_point - self.position)

    # Sprawdza przeciecie promienia z kula
    def trace(self, ray):
        distance = ray.starting_point - self.position
        a = np.dot(ray.direction, ray.direction)
        b = 2 * np.dot(ray.direction, distance)
        c = np.dot(distance, distance) - self.radius ** 2
        d = b ** 2 - 4 * a * c

        if d < 0:
            return (None, None)

        sqrt_d = np.sqrt(d)
        denominator = 1 / (2 * a)

        if d > 0:
            r1 = (-b - sqrt_d) * denominator
            r2 = (-b + sqrt_d) * denominator
            if r1 < EPSILON:
                if r2 < EPSILON:
                    return (None, None)
                r1 = r2
        else:
            r1 = -b * denominator
            if r1 < EPSILON:
                return (None, None)

        cross_point = ray.starting_point + r1 * ray.direction
        return cross_point, r1

# Klasa reprezentujaca kamere (punkt widzenia obserwatora)
class Camera:
    def __init__(self, position=np.array([0, 0, -3]), look_at=np.array([0, 0, 0])):
        self.z_near = 1
        self.pixel_height = 500
        self.pixel_width = 700
        self.povy = 45
        look = normalize(look_at - position)
        self.up = normalize(np.cross(np.cross(look, np.array([0, 1, 0])), look))
        self.position = position
        self.look_at = look_at
        self.direction = normalize(look_at - position)
        aspect = self.pixel_width / self.pixel_height
        povy = self.povy * np.pi / 180
        self.world_height = 2 * np.tan(povy / 2) * self.z_near
        self.world_width = aspect * self.world_height

        center = self.position + self.direction * self.z_near
        width_vector = normalize(np.cross(self.up, self.direction))
        self.translation_vector_x = width_vector * -(self.world_width / self.pixel_width)
        self.translation_vector_y = self.up * -(self.world_height / self.pixel_height)
        self.starting_point = center + width_vector * (self.world_width / 2) + (self.up * self.world_height / 2)

    # Zwraca pozycje danego piksela w przestrzeni 3D
    def get_world_pixel(self, x, y):
        return self.starting_point + self.translation_vector_x * x + self.translation_vector_y * y

# Klasa reprezentujaca cala scene (obiekty, swiatlo, kamera)
class Scene:
    def __init__(self, objects, light, camera):
        self.objects = objects
        self.light = light
        self.camera = camera
        self.ambient = np.array([0.1, 0.1, 0.1])
        self.background = np.array([0, 0, 0])

# Klasa odpowiedzialna za renderowanie sceny
class RayTracer:
    def __init__(self, scene):
        self.scene = scene

    # Generuje koncowy obraz (piksel po pikselu)
    def generate_image(self):
        camera = self.scene.camera
        image = np.zeros((camera.pixel_height, camera.pixel_width, 3))
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                world_pixel = camera.get_world_pixel(x, y)
                direction = normalize(world_pixel - camera.position)
                image[y][x] = self._get_pixel_color(Ray(world_pixel, direction))
        return image

    # Zwraca kolor piksela, na podstawie promienia
    def _get_pixel_color(self, ray):
        obj, distance, cross_point = self._get_closest_object(ray)
        if not obj:
            return self.scene.background
        return obj.get_color(cross_point, ray.direction, self.scene)

    # Znajduje najblizszy obiekt przeciety przez promien
    def _get_closest_object(self, ray):
        closest = None
        min_distance = np.inf
        min_cross_point = None
        for obj in self.scene.objects:
            cross_point, distance = obj.trace(ray)
            if cross_point is not None and distance < min_distance:
                min_distance = distance
                closest = obj
                min_cross_point = cross_point
        return (closest, min_distance, min_cross_point)

# Stworzenie sceny z kilkoma kulami
scene = Scene(
    objects=[
        Sphere(position=np.array([0, 0, 0]), radius=1.5),  # kula glowna
        Sphere(position=np.array([-2.5, -0.5, 0.5]), radius=1.0,
               diffuse=np.array([0.4, 0.3, 0.8]))           # kula blekitna
    ],
    light=Light(position=np.array([3, 2, 5])),
    camera=Camera(position=np.array([0, 0, 5]))
)

# Uruchomienie raytracera i wygenerowanie obrazu
rt = RayTracer(scene)
image = np.clip(rt.generate_image(), 0, 1)

# Wyswietlenie obrazu
plt.imshow(image)
plt.axis('off')
plt.show()
