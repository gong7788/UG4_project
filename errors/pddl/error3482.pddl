(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b4)
		(clear b4)
		(on-table b5)
		(clear b5)
		(on-table b7)
		(clear b7)
		(clear b8)
		(on-table b9)
		(clear b9)
		(in-tower t0)
		(on b0 t0)
		(in-tower b0)
		(on b6 b0)
		(in-tower b6)
		(on b2 b6)
		(in-tower b2)
		(on b3 b2)
		(in-tower b3)
		(on b1 b3)
		(in-tower b1)
		(on b8 b1)
		(in-tower b8)
		(blue b0)
		(yellow b1)
		(yellow b2)
		(blue b2)
		(green b3)
		(red b3)
		(yellow b4)
		(blue b4)
		(green b6)
		(blue b6)
		(green b7)
		(red b7)
		(green b9)
		(green b2)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?y) (or (not (yellow ?y)) (exists (?x) (and (green ?x) (on ?x ?y))))) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (and (not (on b7 b6)) (not (on b3 b6)) (not (on b4 b2)) (not (on b9 b1)) (not (on b9 b8)) (not (on b7 b8)))))
)