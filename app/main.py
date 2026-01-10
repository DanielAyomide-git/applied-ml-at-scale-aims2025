from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import os, requests
from .recommender import recommend_for_user_profile, get_top_popular_movies
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
# Get keys from environment
OMDB_KEY = os.getenv("OMDB_KEY")
TMDB_KEY = os.getenv("TMDB_KEY")

app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    template_path = os.path.join(os.path.dirname(__file__), "templates/index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/search")
def search_movies(q: str):
    url = f"http://www.omdbapi.com/?apikey={OMDB_KEY}&s={q}"
    return requests.get(url).json()

@app.get("/api/movie")
def get_movie(imdb_id: str):
    movie = requests.get(f"http://www.omdbapi.com/?apikey={OMDB_KEY}&i={imdb_id}&plot=full").json()

    # TMDB trailer logic
    tmdb_search = requests.get(
        f"https://api.themoviedb.org/3/find/{imdb_id}",
        params={"api_key": TMDB_KEY, "external_source": "imdb_id"},
    ).json()

    trailer_key = None
    if tmdb_search.get("movie_results"):
        tmdb_id = tmdb_search["movie_results"][0]["id"]
        vids = requests.get(f"https://api.themoviedb.org/3/movie/{tmdb_id}/videos", params={"api_key": TMDB_KEY}).json()
        for v in vids.get("results", []):
            if v["site"] == "YouTube" and v["type"] == "Trailer":
                trailer_key = v["key"]
                break

    return {"movie": movie, "trailer": trailer_key}



@app.post("/api/recommend")
def recommend(payload: dict):
    user_ratings = payload.get("ratings", {})
    if not user_ratings:
        return {"recommendations": []}

    # 1. Get 50 candidates from the math model
    candidates = recommend_for_user_profile(user_ratings, top_n=50)

    detailed_recs = []
    target_count = 10
    
    # --- DYNAMIC GENRE FILTERING ---
    # We fetch the first rated movie to see what "vibe" the user is in
    rated_ids = list(user_ratings.keys())
    is_safe_mode = False
    try:
        # Check the first movie you rated to determine safe mode
        first_rating_res = requests.get(f"http://www.omdbapi.com/?apikey={OMDB_KEY}&i={rated_ids[0]}").json()
        rated_genres = first_rating_res.get("Genre", "")
        if any(g in rated_genres for g in ["Animation", "Family", "Children"]):
            is_safe_mode = True
    except:
        pass

    for c in candidates:
        if len(detailed_recs) >= target_count:
            break
            
        imdb_id = c.get("imdb_id")
        formatted_id = f"tt{str(imdb_id).zfill(7)}" if not str(imdb_id).startswith('tt') else imdb_id

        try:
            res = requests.get(f"http://www.omdbapi.com/?apikey={OMDB_KEY}&i={formatted_id}", timeout=1).json()
            
            if res.get("Response") == "True":
                poster = res.get("Poster")
                genre = res.get("Genre", "")
                
                # Filter 1: Must have a poster
                if not poster or poster == "N/A":
                    continue
                
                # Filter 2: If Safe Mode is on, block unrelated heavy genres
                if is_safe_mode:
                    bad_genres = ["Horror", "Thriller", "Crime", "Drama"]
                    # Only skip if it's ONLY a bad genre (e.g., skip Horror, but keep Adventure/Drama)
                    if any(bg in genre for bg in ["Horror", "Mystery"]):
                        continue
                
                res["imdbID"] = formatted_id 
                detailed_recs.append(res)
        except:
            continue
            
    return {"recommendations": detailed_recs}

@app.get("/api/popular")
def top_50_movies():
    # Changed to 50
    ids = get_top_popular_movies(20)
    popular_list = []
    for imdb_id in ids:
        try:
            res = requests.get(f"http://www.omdbapi.com/?apikey={OMDB_KEY}&i={imdb_id}", timeout=1).json()
            if res.get("Title"):
                popular_list.append(res)
        except:
            continue
    return {"movies": popular_list}