(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b0)
		(clear b0)
		(on-table b1)
		(clear b1)
		(clear b2)
		(on-table b6)
		(clear b6)
		(on-table b7)
		(clear b7)
		(on-table b8)
		(clear b8)
		(on-table b9)
		(clear b9)
		(in-tower t0)
		(on b4 t0)
		(in-tower b4)
		(on b5 b4)
		(in-tower b5)
		(on b3 b5)
		(in-tower b3)
		(on b2 b3)
		(in-tower b2)
		(red b1)
		(green b2)
		(yellow b2)
		(blue b6)
		(blue b7)
		(red b8)
		(blue b8)
		(green b9)
		(blue b9)
		(blue b2)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?y) (or (not (yellow ?y)) (exists (?x) (and (green ?x) (on ?x ?y))))) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (and (not (on b6 b4)) (not (on b8 b4)) (not (on b9 b2)))))
)