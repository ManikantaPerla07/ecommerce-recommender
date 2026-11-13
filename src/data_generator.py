"""
Synthetic E-Commerce Dataset Generator.
Generates realistic product catalogs, user profiles, and chronological user-item
interaction logs exhibiting realistic e-commerce characteristics:
- Power-law (Pareto) item popularity distribution (head vs long-tail items)
- User category preference clustering and varying activity levels
- Multi-action interaction funnel: view -> cart -> purchase -> rating
- Realistic cold-start items and cold-start users
- Strict temporal sequence over 90 days with session tracking
"""
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List, Dict
from src.config import (
    RANDOM_SEED, NUM_USERS, NUM_PRODUCTS, NUM_INTERACTIONS, DAYS_SPAN,
    PRODUCTS_FILE, USERS_FILE, INTERACTIONS_FILE
)

# Seed all generators for 100% deterministic reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CATEGORIES_SPEC = {
    "Electronics": {
        "subcategories": ["Audio", "Computers & Accessories", "Mobile Accessories", "Wearables", "Home Entertainment"],
        "brands": ["Sony", "Apple", "Samsung", "Anker", "Bose", "Logitech", "LG", "Dell", "Sennheiser", "Garmin"],
        "templates": [
            ("Wireless Noise-Cancelling Headphones", "Audio", 199.99, "Over-ear bluetooth active noise cancelling headphones with 30-hour battery life and high-fidelity sound.", ["wireless", "noise-cancelling", "headphones", "bluetooth", "audio", "mic"]),
            ("Smart 4K Ultra HD OLED TV 55-inch", "Home Entertainment", 899.99, "Stunning true-black OLED display with Dolby Vision HDR, HDMI 2.1, and voice assistant.", ["tv", "4k", "oled", "smart tv", "hdr", "gaming"]),
            ("Ergonomic Mechanical Keyboard RGB", "Computers & Accessories", 119.99, "Hot-swappable mechanical switches with tactile feedback, programmable RGB backlight, and wrist rest.", ["keyboard", "mechanical", "ergonomic", "rgb", "gaming", "typing"]),
            ("Portable Power Bank 20000mAh 65W PD", "Mobile Accessories", 49.99, "Fast-charging portable battery bank with dual USB-C ports, compatible with laptops and smartphones.", ["power bank", "charger", "usb-c", "portable", "battery", "fast charge"]),
            ("True Wireless Earbuds with ANC", "Audio", 129.99, "Compact water-resistant earbuds featuring adaptive noise transparency and wireless charging case.", ["earbuds", "wireless", "anc", "bluetooth", "in-ear", "waterproof"]),
            ("Ultra-Wide Curved Gaming Monitor 34-inch", "Computers & Accessories", 449.99, "144Hz refresh rate 1ms response curved monitor with vibrant colors and FreeSync Premium.", ["monitor", "curved", "gaming", "ultrawide", "144hz", "display"]),
            ("USB-C 10-in-1 Triple Display Docking Station", "Computers & Accessories", 79.99, "Multi-port aluminum hub supporting dual 4K HDMI, Gigabit Ethernet, 100W PD pass-through.", ["docking station", "usb-c", "hub", "hdmi", "adapter", "laptop"]),
            ("Smart Fitness GPS Watch with Heart Rate", "Wearables", 179.99, "Multi-sport activity tracking with continuous biometric monitoring, sleep stages, and 7-day battery.", ["smartwatch", "fitness", "gps", "heart rate", "wearable", "sports"]),
            ("Waterproof Portable Bluetooth Speaker", "Audio", 59.99, "Rugged IPX7 waterproof outdoor speaker with deep bass, 360-degree sound, and 12-hour playtime.", ["speaker", "bluetooth", "waterproof", "portable", "outdoor", "bass"]),
            ("Fast Wireless Charging Stand 15W", "Mobile Accessories", 29.99, "Qi-certified upright wireless charger with dual coils for portrait or landscape orientation charging.", ["wireless charger", "qi", "fast charging", "stand", "iphone", "android"])
        ]
    },
    "Fashion": {
        "subcategories": ["Men's Clothing", "Women's Clothing", "Footwear", "Accessories", "Outerwear"],
        "brands": ["Nike", "Adidas", "Levi's", "Zara", "Patagonia", "Ray-Ban", "Uniqlo", "Tommy Hilfiger", "Calvin Klein", "Puma"],
        "templates": [
            ("Classic Oxford 100% Cotton Button-Down Shirt", "Men's Clothing", 45.00, "Timeless tailored fit button-down shirt woven from breathable long-staple cotton.", ["shirt", "cotton", "oxford", "formal", "casual", "menswear"]),
            ("Slim-Fit Stretch Denim Jeans", "Men's Clothing", 59.50, "Durable stretch denim offering modern silhouette with all-day flexible comfort.", ["jeans", "denim", "slim-fit", "pants", "casual", "streetwear"]),
            ("Breathable Lightweight Running Shoes", "Footwear", 85.00, "Responsive foam cushioning with engineered mesh upper for endurance running and daily walks.", ["running shoes", "sneakers", "cushioned", "breathable", "footwear", "sports"]),
            ("Classic Aviator Polarized Sunglasses", "Accessories", 135.00, "Metal frame with 100% UV protection polarized crystal lenses reducing glare.", ["sunglasses", "aviator", "polarized", "uv protection", "eyewear", "accessories"]),
            ("Genuine Full-Grain Leather Bifold Wallet", "Accessories", 38.00, "Handcrafted genuine leather slim wallet with RFID-blocking technology and card slots.", ["wallet", "leather", "bifold", "rfid", "mens", "accessories"]),
            ("Weather-Resistant Packable Windbreaker Jacket", "Outerwear", 75.00, "Ultralight water-repellent shell with hood, zippered pockets, and compact stuff sack.", ["jacket", "windbreaker", "outerwear", "water-resistant", "hooded", "travel"]),
            ("Floral Print Sleeveless Bohemian Midi Dress", "Women's Clothing", 54.00, "Flowing tiered rayon fabric with sweetheart neckline, elastic smocked back, and vibrant prints.", ["dress", "midi", "boho", "floral", "summer", "womenswear"]),
            ("Merino Wool Soft Crewneck Knit Sweater", "Men's Clothing", 68.00, "Naturally thermoregulating fine merino knit with ribbed cuffs and classic crew collar.", ["sweater", "merino wool", "knitwear", "warm", "casual", "pullover"]),
            ("High-Waisted Compressive Workout Leggings", "Women's Clothing", 42.00, "Squat-proof four-way stretch fabric with tummy control waist and dual drop-in phone pockets.", ["leggings", "workout", "yoga pants", "activewear", "high-waisted", "fitness"]),
            ("Canvas Low-Top Casual Skate Sneakers", "Footwear", 49.00, "Vulcanized rubber sole with durable cotton canvas upper and cushioned footbed.", ["sneakers", "skate shoes", "canvas", "low-top", "casual", "shoes"])
        ]
    },
    "Home & Kitchen": {
        "subcategories": ["Kitchen Appliances", "Cookware", "Home Organization", "Bedding & Bath", "Decor"],
        "brands": ["Instant Pot", "Cuisinart", "Lodge", "Dyson", "Ninja", "Philips", "KitchenAid", "Brooklinen", "Simplehuman", "Nespresso"],
        "templates": [
            ("Programmable 7-in-1 Multi-Use Electric Pressure Cooker", "Kitchen Appliances", 89.99, "Stainless steel pressure cooker, slow cooker, rice cooker, steamer, sauté pan, and warmer.", ["pressure cooker", "slow cooker", "instant", "kitchen", "appliances", "cooking"]),
            ("Enameled Cast Iron Dutch Oven 6-Quart", "Cookware", 69.95, "Even heat retention and distribution with durable porcelain enamel finish and self-basting lid.", ["dutch oven", "cast iron", "cookware", "enameled", "baking", "casserole"]),
            ("High-Power Countertop Blender with Pulse", "Kitchen Appliances", 119.00, "1200-watt motor capable of crushing ice, blending smoothies, pureeing soups, and grinding seeds.", ["blender", "smoothie", "food processor", "countertop", "appliances", "kitchen"]),
            ("Robot Vacuum and Mop with LiDAR Navigation", "Home Organization", 299.99, "Smart precision mapping, multi-surface cleaning, auto-carpet boost, and app scheduling.", ["robot vacuum", "smart home", "cleaning", "mop", "lidar", "automation"]),
            ("Ergonomic Shredded Memory Foam Pillow", "Bedding & Bath", 36.99, "Adjustable loft pillow with cooling bamboo-derived rayon cover for neck and shoulder support.", ["pillow", "memory foam", "cooling", "bedding", "sleep", "ergonomic"]),
            ("12-Piece Nonstick Ceramic Cookware Set", "Cookware", 129.99, "PTFE and PFOA free nonstick pots and pans with stay-cool handles and tempered glass lids.", ["cookware set", "pots and pans", "ceramic", "nonstick", "healthy cooking", "kitchen"]),
            ("Organic Bamboo Cutting Board Set with Juice Groove", "Kitchen Appliances", 27.99, "Three versatile kitchen boards carved from sustainable organic bamboo with side handles.", ["cutting board", "bamboo", "kitchen", "prep", "chopping", "wooden"]),
            ("Ultrasonic Essential Oil Aroma Diffuser 500ml", "Decor", 24.99, "Whisper-quiet cool mist humidifier with 7 LED ambient color lights and auto-shutoff timer.", ["diffuser", "aromatherapy", "essential oil", "humidifier", "wellness", "home"]),
            ("Single-Serve Coffee and Espresso Machine", "Kitchen Appliances", 159.00, "Fast heating pod brewing system producing rich crema espresso and premium drip-style cups.", ["coffee maker", "espresso", "pods", "single serve", "cappuccino", "beverages"]),
            ("Stainless Steel Sensor Trash Can 13-Gallon", "Home Organization", 89.99, "Hands-free infrared motion lid with fingerprint-resistant finish and odor filter compartment.", ["trash can", "sensor", "stainless steel", "kitchen", "hygiene", "hands-free"])
        ]
    },
    "Beauty & Personal Care": {
        "subcategories": ["Skincare", "Haircare", "Oral Care", "Body Care", "Fragrance"],
        "brands": ["The Ordinary", "CeraVe", "Oral-B", "Olaplex", "La Roche-Posay", "Neutrogena", "Moroccanoil", "L'Oreal", "Glossier", "Drunk Elephant"],
        "templates": [
            ("Pure Hyaluronic Acid 2% + B5 Hydration Serum", "Skincare", 16.50, "Multi-depth hydration serum replenishing moisture barriers for visibly plumper, smoother skin.", ["serum", "hyaluronic acid", "skincare", "hydration", "anti-aging", "facial"]),
            ("Rechargeable Sonic Electric Toothbrush", "Oral Care", 49.99, "40,000 vibrations per minute with 2-minute smart timer, 4 cleaning modes, and travel case.", ["toothbrush", "electric", "sonic", "dental", "oral care", "whitening"]),
            ("Broad Spectrum Mineral Sunscreen SPF 50", "Skincare", 22.00, "Non-greasy sheer zinc oxide formula offering UVA/UVB protection without white cast.", ["sunscreen", "spf 50", "mineral", "skincare", "sun protection", "face"]),
            ("Intensive Bond Repair Hair Treatment No. 3", "Haircare", 30.00, "Strengthening restorative pre-shampoo treatment reducing breakage and repairing damaged strands.", ["hair treatment", "bond repair", "haircare", "salon", "damage repair", "mask"]),
            ("Gentle Foaming Hydrating Facial Cleanser", "Skincare", 15.00, "Formulated with ceramides and niacinamide to cleanse impurities without stripping natural moisture.", ["cleanser", "face wash", "gentle", "skincare", "dermatologist", "sensitive"]),
            ("Organic Pure Cold-Pressed Argan Oil", "Haircare", 18.99, "Pure nourishing Moroccan oil providing deep moisture for frizzy hair, dry cuticles, and skin.", ["argan oil", "hair oil", "moroccan", "organic", "moisturizer", "shine"]),
            ("Exfoliating 2% BHA Salicylic Acid Liquid Solution", "Skincare", 32.00, "Leave-on exfoliant clearing clogged pores, blackheads, and smoothing uneven texture.", ["exfoliant", "salicylic acid", "bha", "acne", "clear skin", "pores"]),
            ("Nourishing Shea Butter Body Lotion 16oz", "Body Care", 14.50, "Rich hydrating daily cream with raw shea butter and vitamin E for 48-hour moisture barrier.", ["body lotion", "shea butter", "moisturizer", "dry skin", "body care", "cream"]),
            ("Revitalizing Caffeine Eye Cream for Dark Circles", "Skincare", 19.50, "Targeted peptide eye formula reducing puffiness, morning bags, and stubborn under-eye shadows.", ["eye cream", "caffeine", "peptides", "dark circles", "anti-aging", "eyes"]),
            ("Volumizing Biotin and Keratin Shampoo Set", "Haircare", 26.00, "Sulfate-free thickening shampoo and conditioner strengthening thin hair and adding body.", ["shampoo", "conditioner", "biotin", "keratin", "volumizing", "haircare"])
        ]
    },
    "Sports & Fitness": {
        "subcategories": ["Fitness Equipment", "Yoga & Pilates", "Outdoor Recreation", "Active Accessories", "Recovery"],
        "brands": ["Bowflex", "Manduka", "Hydro Flask", "Theragun", "Under Armour", "Lululemon", "Fitbit", "CamelBak", "Everlast", "Gaiam"],
        "templates": [
            ("Adjustable Quick-Select Dumbbell (Pair 5-52.5 lbs)", "Fitness Equipment", 299.00, "Space-saving selector dial dumbbells replacing 15 sets of weights for strength training.", ["dumbbells", "adjustable weights", "strength training", "home gym", "fitness", "workout"]),
            ("Extra-Thick High-Density Non-Slip Yoga Mat", "Yoga & Pilates", 38.00, "Eco-friendly textured TPE surface with body alignment lines and convenient carrying strap.", ["yoga mat", "pilates", "non-slip", "exercise mat", "fitness", "alignment"]),
            ("Vacuum Insulated Stainless Steel Water Bottle 32oz", "Active Accessories", 34.95, "Double-wall vacuum insulation keeping beverages ice-cold for 24 hours with leak-proof straw lid.", ["water bottle", "insulated", "stainless steel", "hydro", "sports", "hydration"]),
            ("Deep Tissue Percussion Muscle Massage Gun", "Recovery", 79.99, "Quiet brushless motor with 6 interchangeable massage heads and 30 speed levels for myofascial relief.", ["massage gun", "recovery", "deep tissue", "muscle soreness", "therapy", "fitness"]),
            ("Heavy-Duty Resistance Loop Bands Set of 5", "Fitness Equipment", 19.99, "Natural latex exercise bands with varied resistance levels for glute activation and physical therapy.", ["resistance bands", "loop bands", "workout", "latex", "strength", "physical therapy"]),
            ("Pro Speed Jump Rope with Ball Bearings", "Fitness Equipment", 14.99, "Tangle-free adjustable steel wire rope for high-intensity cardio, double unders, and boxing.", ["jump rope", "speed rope", "cardio", "boxing", "fitness", "endurance"]),
            ("High-Density Trigger Point Foam Roller", "Recovery", 22.50, "Grid matrix pattern foam roller targeting tight muscles, back tension, and post-workout soreness.", ["foam roller", "recovery", "trigger point", "back pain", "mobility", "fitness"]),
            ("Tactical Molle Hydration Backpack 2L Water Bladder", "Outdoor Recreation", 44.99, "Lightweight breathable pack designed for day hiking, trail running, biking, and trekking.", ["backpack", "hydration pack", "hiking", "outdoor", "trail running", "camping"]),
            ("Gym Grip Padded Weightlifting Gloves with Wrist Wraps", "Active Accessories", 18.00, "Anti-slip silicone palm padding protecting hands from calluses with integrated wrist support.", ["weightlifting gloves", "gym gloves", "wrist support", "grip", "crossfit", "training"]),
            ("Electronic Hand Grip Strengthener and Dynamometer", "Fitness Equipment", 24.99, "Digital grip strength tester with clear LCD screen measuring force up to 198 lbs.", ["grip strengthener", "dynamometer", "forearm", "hand strength", "fitness", "tester"])
        ]
    },
    "Books & Stationery": {
        "subcategories": ["Business & Tech Books", "Self-Help", "Journals & Planners", "Writing Instruments", "Office Supplies"],
        "brands": ["Penguin", "O'Reilly", "Moleskine", "Pilot", "Rhodia", "Faber-Castell", "HarperCollins", "Midori", "Oxford", "Leuchtturm1917"],
        "templates": [
            ("Designing Data-Intensive Applications Book", "Business & Tech Books", 42.00, "The definitive guide to the architecture and principles of reliable, scalable, and maintainable systems.", ["book", "software engineering", "data", "architecture", "distributed systems", "tech"]),
            ("Atomic Habits: An Easy & Proven Way to Build Good Habits", "Self-Help", 18.00, "Global bestseller on behavioral change, micro-habits, productivity, and personal development.", ["book", "atomic habits", "productivity", "self-help", "psychology", "habits"]),
            ("Deep Work: Rules for Focused Success in a Distracted World", "Self-Help", 17.50, "Essential strategies for cultivating deep concentration, eliminating digital noise, and peak performance.", ["book", "deep work", "focus", "productivity", "career", "study"]),
            ("Classic Hardcover Dotted Grid Notebook / Journal", "Journals & Planners", 19.95, "Thick 120gsm ink-proof numbered pages with expandable pocket, ribbon bookmark, and elastic closure.", ["notebook", "journal", "dotted", "bullet journal", "stationery", "hardcover"]),
            ("Fine Point Premium Retractable Gel Ink Pens (Pack of 12)", "Writing Instruments", 14.50, "Smooth quick-drying black ink pens with rubber grip preventing smearing for journaling and note-taking.", ["gel pens", "pens", "stationery", "writing", "office", "quick-drying"]),
            ("Undated Daily Goal Planner & Productivity Journal", "Journals & Planners", 22.00, "Structured day-by-day task planning, priority matrices, habit tracking, and quarterly reflection.", ["planner", "productivity", "daily planner", "organizer", "journal", "goals"]),
            ("Architectural Drafting Clutch Mechanical Pencil 0.5mm", "Writing Instruments", 16.99, "Precision metal barrel with knurled grip, brass mechanism, and replaceable lead indicator.", ["mechanical pencil", "drafting", "sketching", "stationery", "design", "pencil"]),
            ("Clean Code: A Handbook of Agile Software Craftsmanship", "Business & Tech Books", 39.99, "Foundational best practices on code readability, refactoring, testing, and object-oriented design.", ["book", "programming", "clean code", "software", "agile", "best practices"]),
            ("Self-Healing Rotating Cutting Mat & Rotary Cutter Set", "Office Supplies", 26.50, "Double-sided grid mat protecting tabletops with sharp 45mm fabric cutter for crafts and papers.", ["cutting mat", "rotary cutter", "crafting", "office", "stationery", "tools"]),
            ("Vintage Wood Dip Fountain Pen with 6 Calligraphy Nibs", "Writing Instruments", 21.00, "Traditional calligraphy set with handcrafted wooden nib holder, assorted stainless nibs, and black ink.", ["fountain pen", "calligraphy", "lettering", "art", "stationery", "gift"])
        ]
    }
}


def generate_catalog(num_products: int = NUM_PRODUCTS) -> pd.DataFrame:
    """Generate product catalog across balanced e-commerce categories."""
    products = []
    categories = list(CATEGORIES_SPEC.keys())
    items_per_cat = num_products // len(categories)
    remainder = num_products % len(categories)

    prod_idx = 1
    for cat_idx, category in enumerate(categories):
        quota = items_per_cat + (1 if cat_idx < remainder else 0)
        spec = CATEGORIES_SPEC[category]
        templates = spec["templates"]
        brands = spec["brands"]
        subcats = spec["subcategories"]

        for i in range(quota):
            template = templates[i % len(templates)]
            brand = brands[i % len(brands)]
            subcat = template[1] if len(templates) > 0 else random.choice(subcats)
            base_title, _, base_price, base_desc, base_tags = template

            # Add variation if cycling through templates
            variation = f" (Edition {i // len(templates) + 1})" if i >= len(templates) else ""
            title = f"{brand} {base_title}{variation}"
            price = round(max(5.99, base_price * (1.0 + np.random.uniform(-0.15, 0.20))), 2)

            # Rating distribution centered around 4.2 with slight variance
            rating_avg = round(float(np.clip(np.random.normal(4.25, 0.45), 2.5, 5.0)), 2)
            rating_count = int(np.random.pareto(a=1.5) * 50) + 10

            # Tag enrichment
            tags = list(set(base_tags + [category.lower(), subcat.lower(), brand.lower()]))

            # Product release date (simulate cold-start items introduced in the last 15% of the 90-day window)
            # 8% of products are brand new cold-start items created in days 75-90
            is_cold_start = (prod_idx % 12 == 0)
            if is_cold_start:
                days_ago = np.random.uniform(1, 10)  # Very recent
            else:
                days_ago = np.random.uniform(15, DAYS_SPAN)

            created_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")

            products.append({
                "product_id": f"PROD_{prod_idx:04d}",
                "title": title,
                "category": category,
                "subcategory": subcat,
                "brand": brand,
                "price": price,
                "rating_avg": rating_avg,
                "rating_count": rating_count,
                "description": f"{brand} presents {title}. {base_desc}",
                "tags": ", ".join(tags),
                "is_cold_start_candidate": is_cold_start,
                "created_at": created_date
            })
            prod_idx += 1

    df_products = pd.DataFrame(products)
    return df_products


def generate_users(num_users: int = NUM_USERS) -> pd.DataFrame:
    """Generate user profiles with demographic segments and category preferences."""
    users = []
    categories = list(CATEGORIES_SPEC.keys())
    age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]
    genders = ["Female", "Male", "Non-Binary", "Unspecified"]

    for u_idx in range(1, num_users + 1):
        primary_cat = random.choice(categories)
        secondary_cat = random.choice([c for c in categories if c != primary_cat])

        # User activity tier: Power users (15%), Regular users (60%), Casual/Cold-start users (25%)
        rand_tier = random.random()
        if rand_tier < 0.15:
            activity_tier = "power"
            expected_interactions = random.randint(25, 60)
        elif rand_tier < 0.75:
            activity_tier = "regular"
            expected_interactions = random.randint(8, 22)
        else:
            activity_tier = "casual"  # includes cold-start users with <= 3 interactions
            expected_interactions = random.randint(1, 4)

        # Signup date
        is_new_user = (u_idx % 10 == 0)  # 10% signed up very recently (cold-start users)
        if is_new_user:
            signup_days_ago = random.uniform(1, 10)
        else:
            signup_days_ago = random.uniform(12, DAYS_SPAN + 10)

        signup_date = (datetime.now() - timedelta(days=signup_days_ago)).strftime("%Y-%m-%d %H:%M:%S")

        users.append({
            "user_id": f"USER_{u_idx:04d}",
            "age_group": random.choice(age_groups),
            "gender": random.choice(genders),
            "primary_category": primary_cat,
            "secondary_category": secondary_cat,
            "activity_tier": activity_tier,
            "expected_interactions": expected_interactions,
            "is_cold_start_candidate": is_new_user,
            "signup_date": signup_date
        })

    df_users = pd.DataFrame(users)
    return df_users


def generate_interactions(
    df_products: pd.DataFrame,
    df_users: pd.DataFrame,
    target_interactions: int = NUM_INTERACTIONS
) -> pd.DataFrame:
    """
    Generate realistic user-item interaction logs with chronological integrity.
    Follows power-law item popularity, user affinity, funnel progression, and session grouping.
    """
    categories = list(CATEGORIES_SPEC.keys())
    products_by_category = {cat: df_products[df_products["category"] == cat]["product_id"].tolist() for cat in categories}
    all_product_ids = df_products["product_id"].tolist()

    # Create Zipf/power-law sampling probabilities for products
    # Top 20% products get ~70% of interactions
    product_weights = np.random.pareto(a=1.8, size=len(all_product_ids)) + 0.1
    product_weights /= product_weights.sum()
    prod_weight_map = dict(zip(all_product_ids, product_weights))

    interactions = []
    base_start_time = datetime.now() - timedelta(days=DAYS_SPAN)

    for _, user in df_users.iterrows():
        u_id = user["user_id"]
        tier = user["activity_tier"]
        p_cat = user["primary_category"]
        s_cat = user["secondary_category"]
        signup_dt = datetime.strptime(user["signup_date"], "%Y-%m-%d %H:%M:%S")

        n_events = user["expected_interactions"]

        # User activity occurs between signup_date and now
        time_available_days = max(1.0, (datetime.now() - signup_dt).total_seconds() / 86400.0)

        # Generate interaction sessions
        num_sessions = max(1, n_events // 3)
        for s_idx in range(num_sessions):
            session_id = f"SESS_{u_id}_{s_idx:03d}"
            session_day_offset = random.uniform(0, time_available_days)
            session_time = signup_dt + timedelta(days=session_day_offset)

            # Session event count
            sess_event_count = max(1, n_events // num_sessions)

            # Determine product focus for this session (affinity biased)
            if random.random() < 0.65:
                cat_choice = p_cat
            elif random.random() < 0.85:
                cat_choice = s_cat
            else:
                cat_choice = random.choice(categories)

            candidate_prods = products_by_category[cat_choice]
            # Sample candidate product biased by power-law weight
            c_weights = [prod_weight_map[p] for p in candidate_prods]
            c_weights = np.array(c_weights) / sum(c_weights)

            # Choose 1 or 2 focal products in this session
            focal_prods = np.random.choice(candidate_prods, size=min(2, len(candidate_prods)), replace=False, p=c_weights)

            curr_event_time = session_time
            for p_id in focal_prods:
                # E-commerce funnel sequence for this product
                # 1. View is almost guaranteed
                interactions.append({
                    "user_id": u_id,
                    "product_id": p_id,
                    "event_type": "view",
                    "rating": None,
                    "session_id": session_id,
                    "timestamp": curr_event_time.strftime("%Y-%m-%d %H:%M:%S")
                })

                # 2. Add-to-cart occurs with 35% probability after view
                if random.random() < 0.35:
                    curr_event_time += timedelta(minutes=random.randint(1, 10))
                    interactions.append({
                        "user_id": u_id,
                        "product_id": p_id,
                        "event_type": "cart",
                        "rating": None,
                        "session_id": session_id,
                        "timestamp": curr_event_time.strftime("%Y-%m-%d %H:%M:%S")
                    })

                    # 3. Purchase occurs with 50% probability after cart
                    if random.random() < 0.50:
                        curr_event_time += timedelta(minutes=random.randint(2, 15))
                        interactions.append({
                            "user_id": u_id,
                            "product_id": p_id,
                            "event_type": "purchase",
                            "rating": None,
                            "session_id": session_id,
                            "timestamp": curr_event_time.strftime("%Y-%m-%d %H:%M:%S")
                        })

                        # 4. Rating occurs with 40% probability after purchase
                        if random.random() < 0.40:
                            curr_event_time += timedelta(days=random.uniform(1, 5))
                            rating_val = int(np.clip(np.random.choice([5, 4, 3, 2, 1], p=[0.55, 0.25, 0.12, 0.05, 0.03]), 1, 5))
                            interactions.append({
                                "user_id": u_id,
                                "product_id": p_id,
                                "event_type": "rating",
                                "rating": rating_val,
                                "session_id": session_id,
                                "timestamp": curr_event_time.strftime("%Y-%m-%d %H:%M:%S")
                            })

                curr_event_time += timedelta(minutes=random.randint(2, 8))

    df_interactions = pd.DataFrame(interactions)

    # Sort strictly by timestamp to maintain chronological integrity
    df_interactions["timestamp"] = pd.to_datetime(df_interactions["timestamp"])
    df_interactions = df_interactions.sort_values("timestamp").reset_index(drop=True)
    df_interactions["timestamp"] = df_interactions["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Add interaction ID
    df_interactions.insert(0, "interaction_id", [f"INT_{i:06d}" for i in range(1, len(df_interactions) + 1)])

    return df_interactions


def run_data_generation() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute complete reproducible generation and save datasets to data/raw."""
    print("=" * 60)
    print("STARTING REPRODUCIBLE DATA GENERATION")
    print(f"Target: {NUM_USERS} users, {NUM_PRODUCTS} products, ~{NUM_INTERACTIONS} interactions")
    print(f"Random Seed: {RANDOM_SEED}")
    print("=" * 60)

    df_products = generate_catalog(NUM_PRODUCTS)
    df_users = generate_users(NUM_USERS)
    df_interactions = generate_interactions(df_products, df_users, NUM_INTERACTIONS)

    # Save to disk
    df_products.to_csv(PRODUCTS_FILE, index=False)
    df_users.to_csv(USERS_FILE, index=False)
    df_interactions.to_csv(INTERACTIONS_FILE, index=False)

    print(f"[OK] Generated {len(df_products)} products -> {PRODUCTS_FILE}")
    print(f"[OK] Generated {len(df_users)} users -> {USERS_FILE}")
    print(f"[OK] Generated {len(df_interactions)} interactions -> {INTERACTIONS_FILE}")
    print(f"     Time range: {df_interactions['timestamp'].min()} to {df_interactions['timestamp'].max()}")
    print("=" * 60)

    return df_products, df_users, df_interactions


if __name__ == "__main__":
    run_data_generation()
