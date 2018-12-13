(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b0)
		(clear b0)
		(on-table b5)
		(clear b5)
		(clear b7)
		(in-tower t0)
		(on b9 t0)
		(in-tower b9)
		(on b8 b9)
		(in-tower b8)
		(on b6 b8)
		(in-tower b6)
		(on b4 b6)
		(in-tower b4)
		(on b3 b4)
		(in-tower b3)
		(on b2 b3)
		(in-tower b2)
		(on b1 b2)
		(in-tower b1)
		(on b7 b1)
		(in-tower b7)
		(green b0)
		(yellow b0)
		(red b0)
		(blue b0)
		(green b1)
		(yellow b1)
		(red b1)
		(blue b1)
		(green b2)
		(yellow b2)
		(red b2)
		(blue b2)
		(green b3)
		(yellow b3)
		(red b3)
		(blue b3)
		(green b4)
		(yellow b4)
		(red b4)
		(blue b4)
		(green b5)
		(yellow b5)
		(red b5)
		(blue b5)
		(green b6)
		(yellow b6)
		(red b6)
		(blue b6)
		(green b7)
		(yellow b7)
		(red b7)
		(blue b7)
		(green b8)
		(yellow b8)
		(red b8)
		(blue b8)
		(green b9)
		(yellow b9)
		(blue b9)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?y) (or (not (yellow ?y)) (exists (?x) (and (green ?x) (on ?x ?y))))) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (and (not (on b7 b8)) (not (on b7 b6)) (not (on b5 b6)) (not (on b7 b4)) (not (on b5 b4)) (not (on b7 b3)) (not (on b5 b3)) (not (on b7 b2)) (not (on b5 b2)) (not (on b5 b7)))))
)