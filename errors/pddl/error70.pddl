(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b0)
		(clear b0)
		(on-table b1)
		(clear b1)
		(on-table b2)
		(clear b2)
		(on-table b3)
		(clear b3)
		(on-table b5)
		(clear b5)
		(on-table b8)
		(clear b8)
		(clear b9)
		(in-tower t0)
		(on b4 t0)
		(in-tower b4)
		(on b7 b4)
		(in-tower b7)
		(on b6 b7)
		(in-tower b6)
		(on b9 b6)
		(in-tower b9)
		(red b1)
		(red b2)
		(green b3)
		(blue b4)
		(green b5)
		(green b8)
		(blue b9)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?y) (or (not (yellow ?y)) (exists (?x) (and (green ?x) (on ?x ?y))))) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (and (not (on b2 b6)) (not (on b1 b6)) (not (on b8 b9)) (not (on b5 b9)) (not (on b3 b9)))))
)